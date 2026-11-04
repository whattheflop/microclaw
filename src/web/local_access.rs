use super::*;
use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;

pub(super) fn enabled(config: &Config) -> bool {
    web_channel_bool(config, "loopback_no_auth", false)
}

pub(super) fn valid_bind(config: &Config) -> bool {
    !enabled(config) || config.web_host == "127.0.0.1"
}

pub(super) fn identity() -> AuthIdentity {
    AuthIdentity {
        actor: "localhost-operator".into(),
        scopes: vec![
            "operator.read".into(),
            "operator.write".into(),
            "operator.admin".into(),
            "operator.approvals".into(),
        ],
    }
}

pub(super) async fn prepare(state: &Arc<AppState>) -> Result<(), String> {
    if !valid_bind(&state.config) {
        return Err("loopback_no_auth requires web_host: 127.0.0.1".into());
    }
    if enabled(&state.config) {
        call_blocking(state.db.clone(), |db| {
            db.clear_auth_password_hash()?;
            db.revoke_all_auth_sessions()?;
            Ok(())
        })
        .await
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub(super) async fn guard(State(state): State<WebState>, request: Request, next: Next) -> Response {
    let config = &state.app_state.config;
    if !enabled(config) {
        return next.run(request).await;
    }
    if !valid_bind(config) {
        return StatusCode::FORBIDDEN.into_response();
    }
    let headers = request.headers();
    let host = headers
        .get("host")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if host != format!("127.0.0.1:{}", config.web_port)
        && host != format!("localhost:{}", config.web_port)
    {
        return StatusCode::FORBIDDEN.into_response();
    }
    let origin = headers.get("origin").and_then(|v| v.to_str().ok());
    let expected = format!("http://{host}");
    if origin.is_some_and(|value| value != expected)
        || headers
            .get("sec-fetch-site")
            .is_some_and(|v| v == "cross-site")
    {
        return StatusCode::FORBIDDEN.into_response();
    }
    if !matches!(
        *request.method(),
        axum::http::Method::GET | axum::http::Method::HEAD | axum::http::Method::OPTIONS
    ) && origin.is_none()
        && auth_token_from_headers(headers).is_none()
    {
        return StatusCode::FORBIDDEN.into_response();
    }
    if matches!(
        request.uri().path(),
        "/api/auth/login" | "/api/auth/password" | "/api/auth/logout"
    ) {
        return (
            StatusCode::CONFLICT,
            "Password login is disabled for localhost access",
        )
            .into_response();
    }
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::web::test_prelude::*;
    use axum::http::Request;

    fn app() -> Arc<AppState> {
        let mut cfg = test_config_template();
        cfg.web_port = 19401;
        cfg.a2a.enabled = true;
        cfg.a2a.shared_tokens = vec!["fixture-peer-token".into()];
        let mut web = serde_yaml::Mapping::new();
        web.insert("loopback_no_auth".into(), true.into());
        cfg.channels
            .insert("web".into(), serde_yaml::Value::Mapping(web));
        test_state_with_config(Box::new(DummyLlm), cfg)
    }
    #[tokio::test]
    async fn loopback_access_removes_password_and_opens_ui_without_credentials() {
        let state = app();
        let db = state.db.clone();
        call_blocking(db.clone(), |db| {
            db.upsert_auth_password_hash(&make_password_hash("fixture-only"))
        })
        .await
        .unwrap();
        prepare(&state).await.unwrap();
        assert!(call_blocking(db, |db| db.get_auth_password_hash())
            .await
            .unwrap()
            .is_none());
        let web = test_web_state_from_app_state(state, WebLimits::default());
        let router = build_router(web);
        let response = router
            .oneshot(
                Request::builder()
                    .uri("/api/auth/status")
                    .header("host", "127.0.0.1:19401")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 10000)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["authenticated"], true);
        assert_eq!(json["password_required"], false);
        assert_eq!(json["has_password"], false);
    }
    #[tokio::test]
    async fn loopback_access_rejects_host_origin_and_unauthenticated_a2a() {
        let state = app();
        let router = build_router(test_web_state_from_app_state(state, WebLimits::default()));
        for (host, origin, method, path, expected) in [
            (
                "remote.example:19401",
                None,
                "GET",
                "/api/sessions",
                StatusCode::FORBIDDEN,
            ),
            (
                "127.0.0.1:19401",
                Some("https://remote.example"),
                "GET",
                "/api/sessions",
                StatusCode::FORBIDDEN,
            ),
            (
                "127.0.0.1:19401",
                None,
                "POST",
                "/api/reset",
                StatusCode::FORBIDDEN,
            ),
            (
                "127.0.0.1:19401",
                Some("http://127.0.0.1:19401"),
                "POST",
                "/api/auth/password",
                StatusCode::CONFLICT,
            ),
            (
                "localhost:19401",
                None,
                "GET",
                "/api/sessions",
                StatusCode::OK,
            ),
        ] {
            let mut request = Request::builder()
                .method(method)
                .uri(path)
                .header("host", host);
            if let Some(origin) = origin {
                request = request.header("origin", origin);
            }
            assert_eq!(
                router
                    .clone()
                    .oneshot(request.body(Body::empty()).unwrap())
                    .await
                    .unwrap()
                    .status(),
                expected
            );
        }
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/a2a/message")
                    .header("host", "127.0.0.1:19401")
                    .header("origin", "http://127.0.0.1:19401")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"message":"must not run"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
    #[tokio::test]
    async fn default_access_still_requires_authentication() {
        let state = test_state(Box::new(DummyLlm));
        call_blocking(state.db.clone(), |db| {
            db.upsert_auth_password_hash(&make_password_hash("fixture-only"))
        })
        .await
        .unwrap();
        let router = build_router(test_web_state_from_app_state(state, WebLimits::default()));
        let response = router
            .oneshot(
                Request::builder()
                    .uri("/api/sessions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
    #[test]
    fn loopback_access_refuses_public_bind_and_is_opt_in() {
        let mut config = test_config_template();
        assert!(!enabled(&config));
        let mut web = serde_yaml::Mapping::new();
        web.insert("loopback_no_auth".into(), true.into());
        config
            .channels
            .insert("web".into(), serde_yaml::Value::Mapping(web));
        assert!(valid_bind(&config));
        config.web_host = "0.0.0.0".into();
        assert!(!valid_bind(&config));
    }
}

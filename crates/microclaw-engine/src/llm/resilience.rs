use super::*;

/// Number of consecutive primary-model failures before the breaker opens.
pub(crate) const BREAKER_FAILURE_THRESHOLD: u32 = 4;

/// How long the breaker stays open (routing straight to the fallback) before it
/// lets a primary probe through again.
pub(crate) const BREAKER_COOLDOWN_SECS: u64 = 30;

/// A simple consecutive-failure circuit breaker for the primary model. When the
/// primary fails `threshold` times in a row the breaker opens for `cooldown`,
/// during which calls skip the primary and go straight to the fallback; after
/// the cooldown a single primary probe is allowed through (closing the breaker
/// on success, reopening on failure).
/// Process-wide provider failover stats, aggregated across every
/// `ResilientProvider` instance (main + aux slots). Powers `/status`,
/// the governance snapshot, and the "provider down" alert class.
pub(crate) static FAILOVER_TOTAL: AtomicU64 = AtomicU64::new(0);

pub(crate) static FAILURES_CONSECUTIVE: AtomicU64 = AtomicU64::new(0);

pub(crate) static BREAKER_OPEN_UNTIL_EPOCH_MS: AtomicU64 = AtomicU64::new(0);

pub(crate) static LAST_FAILURE_EPOCH_MS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Debug, Clone, Serialize)]
pub struct ProviderFailoverSnapshot {
    /// Calls that were served by the fallback model since process start.
    pub total_fallbacks: u64,
    /// Provider failures since the last success (approximate: shared across
    /// provider instances, reset by a success on any of them).
    pub consecutive_failures: u64,
    /// True while any circuit breaker's cooldown window is still open. The
    /// window expires on its own and is NOT cleared by a success on a
    /// different provider instance (an aux-slot success must not mask a
    /// down primary), so this can briefly over-report a breaker that
    /// recovered early — a conservative, fail-safe bias.
    pub breaker_open: bool,
    pub last_failure_epoch_ms: Option<u64>,
}

pub fn provider_failover_snapshot() -> ProviderFailoverSnapshot {
    let last = LAST_FAILURE_EPOCH_MS.load(Ordering::Relaxed);
    ProviderFailoverSnapshot {
        total_fallbacks: FAILOVER_TOTAL.load(Ordering::Relaxed),
        consecutive_failures: FAILURES_CONSECUTIVE.load(Ordering::Relaxed),
        breaker_open: BREAKER_OPEN_UNTIL_EPOCH_MS.load(Ordering::Relaxed) > epoch_ms(),
        last_failure_epoch_ms: if last == 0 { None } else { Some(last) },
    }
}

pub(crate) fn note_fallback_used() {
    FAILOVER_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub(crate) struct CircuitBreaker {
    failures: AtomicU32,
    open_until: Mutex<Option<Instant>>,
    threshold: u32,
    cooldown: Duration,
}

impl CircuitBreaker {
    pub(crate) fn new(threshold: u32, cooldown: Duration) -> Self {
        Self {
            failures: AtomicU32::new(0),
            open_until: Mutex::new(None),
            threshold,
            cooldown,
        }
    }

    pub(crate) fn is_open(&self, now: Instant) -> bool {
        match *self.open_until.lock().unwrap_or_else(|p| p.into_inner()) {
            Some(until) => now < until,
            None => false,
        }
    }

    pub(crate) fn record_success(&self) {
        self.failures.store(0, Ordering::Relaxed);
        if let Ok(mut guard) = self.open_until.lock() {
            *guard = None;
        }
        // Reset only this instance's contribution to the shared consecutive
        // counter. The global breaker-open window is deliberately NOT
        // cleared here: a success on this (possibly aux) instance must not
        // mask a different instance's still-open primary breaker; the window
        // expires on its own timeline.
        FAILURES_CONSECUTIVE.store(0, Ordering::Relaxed);
    }

    pub(crate) fn record_failure(&self, now: Instant) {
        let n = self.failures.fetch_add(1, Ordering::Relaxed) + 1;
        FAILURES_CONSECUTIVE.fetch_add(1, Ordering::Relaxed);
        LAST_FAILURE_EPOCH_MS.store(epoch_ms(), Ordering::Relaxed);
        if n >= self.threshold {
            if let Ok(mut guard) = self.open_until.lock() {
                *guard = Some(now + self.cooldown);
            }
            // Extend the shared window to the furthest-out cooldown rather
            // than overwriting, so concurrent breakers can't shorten it.
            BREAKER_OPEN_UNTIL_EPOCH_MS.fetch_max(
                epoch_ms() + self.cooldown.as_millis() as u64,
                Ordering::Relaxed,
            );
        }
    }
}

/// Decorator that adds a circuit breaker + model fallback around any
/// `LlmProvider`. Non-streaming calls fall back to the configured model after a
/// primary failure (or immediately while the breaker is open). Streaming calls
/// may fall back only before emitting text; mid-stream retries would duplicate
/// already-emitted output.
pub struct ResilientProvider {
    inner: Box<dyn LlmProvider>,
    primary_model: String,
    fallback_model: String,
    breaker: CircuitBreaker,
}

impl ResilientProvider {
    pub(crate) fn new(
        inner: Box<dyn LlmProvider>,
        primary_model: String,
        fallback_model: String,
    ) -> Self {
        Self {
            inner,
            primary_model,
            fallback_model,
            breaker: CircuitBreaker::new(
                BREAKER_FAILURE_THRESHOLD,
                Duration::from_secs(BREAKER_COOLDOWN_SECS),
            ),
        }
    }

    async fn call(
        &self,
        system: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        model_override: Option<&str>,
    ) -> Result<MessagesResponse, MicroClawError> {
        let now = Instant::now();
        // Breaker open and the caller didn't pin a model → fail fast to fallback.
        if (model_override.is_none() || model_override == Some(self.primary_model.as_str()))
            && self.breaker.is_open(now)
        {
            warn!(
                "LLM circuit breaker open; routing to fallback model {}",
                self.fallback_model
            );
            note_fallback_used();
            return self
                .inner
                .send_message_with_model(system, messages, tools, Some(&self.fallback_model))
                .await;
        }
        match self
            .inner
            .send_message_with_model(system, messages.clone(), tools.clone(), model_override)
            .await
        {
            Ok(r) => {
                self.breaker.record_success();
                Ok(r)
            }
            Err(e) => {
                self.breaker.record_failure(now);
                // Try the fallback once, unless the caller already asked for it.
                if model_override != Some(self.fallback_model.as_str()) {
                    warn!(
                        "primary LLM call failed ({e}); retrying with fallback model {}",
                        self.fallback_model
                    );
                    note_fallback_used();
                    match self
                        .inner
                        .send_message_with_model(
                            system,
                            messages,
                            tools,
                            Some(&self.fallback_model),
                        )
                        .await
                    {
                        Ok(r) => Ok(r),
                        Err(e2) => {
                            warn!("fallback model also failed: {e2}");
                            Err(e)
                        }
                    }
                } else {
                    Err(e)
                }
            }
        }
    }

    async fn call_stream(
        &self,
        system: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        text_tx: Option<&UnboundedSender<String>>,
        model_override: Option<&str>,
    ) -> Result<MessagesResponse, MicroClawError> {
        let now = Instant::now();
        if (model_override.is_none() || model_override == Some(self.primary_model.as_str()))
            && self.breaker.is_open(now)
        {
            warn!(
                "LLM circuit breaker open; streaming via fallback model {}",
                self.fallback_model
            );
            note_fallback_used();
            return self
                .inner
                .send_message_stream_with_model(
                    system,
                    messages,
                    tools,
                    text_tx,
                    Some(&self.fallback_model),
                )
                .await;
        }
        let mut emitted_text = false;
        let attempt = if let Some(target) = text_tx {
            let (attempt_tx, mut attempt_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
            let attempt = {
                let call = self.inner.send_message_stream_with_model(
                    system,
                    messages.clone(),
                    tools.clone(),
                    Some(&attempt_tx),
                    model_override,
                );
                tokio::pin!(call);
                loop {
                    tokio::select! {
                        result = &mut call => break result,
                        Some(delta) = attempt_rx.recv() => {
                            emitted_text |= !delta.is_empty();
                            let _ = target.send(delta);
                        }
                    }
                }
            };
            // A provider may enqueue a delta and fail in the same poll. Drain it
            // before deciding whether retrying could duplicate visible output.
            while let Ok(delta) = attempt_rx.try_recv() {
                emitted_text |= !delta.is_empty();
                let _ = target.send(delta);
            }
            attempt
        } else {
            self.inner
                .send_message_stream_with_model(
                    system,
                    messages.clone(),
                    tools.clone(),
                    None,
                    model_override,
                )
                .await
        };
        match attempt {
            Ok(r) => {
                self.breaker.record_success();
                Ok(r)
            }
            Err(e) => {
                self.breaker.record_failure(now);
                if !emitted_text && model_override != Some(self.fallback_model.as_str()) {
                    warn!(
                        "primary stream failed before text ({e}); retrying with fallback model {}",
                        self.fallback_model
                    );
                    note_fallback_used();
                    match self
                        .inner
                        .send_message_stream_with_model(
                            system,
                            messages,
                            tools,
                            text_tx,
                            Some(&self.fallback_model),
                        )
                        .await
                    {
                        Ok(response) => return Ok(response),
                        Err(fallback_error) => {
                            warn!("fallback stream also failed: {fallback_error}")
                        }
                    }
                }
                Err(e)
            }
        }
    }
}

#[async_trait]
impl LlmProvider for ResilientProvider {
    async fn send_message(
        &self,
        system: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<MessagesResponse, MicroClawError> {
        self.call(system, messages, tools, None).await
    }

    async fn send_message_with_model(
        &self,
        system: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        model_override: Option<&str>,
    ) -> Result<MessagesResponse, MicroClawError> {
        self.call(system, messages, tools, model_override).await
    }

    async fn send_message_stream(
        &self,
        system: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        text_tx: Option<&UnboundedSender<String>>,
    ) -> Result<MessagesResponse, MicroClawError> {
        self.call_stream(system, messages, tools, text_tx, None)
            .await
    }

    async fn send_message_stream_with_model(
        &self,
        system: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        text_tx: Option<&UnboundedSender<String>>,
        model_override: Option<&str>,
    ) -> Result<MessagesResponse, MicroClawError> {
        self.call_stream(system, messages, tools, text_tx, model_override)
            .await
    }
}

// ---------------------------------------------------------------------------
// Anthropic provider
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::llm::test_prelude::*;
    use std::sync::Arc;

    #[derive(Clone, Copy)]
    enum Primary {
        Success,
        FailBeforeText,
        FailAfterText,
    }

    struct FakeProvider {
        primary: Primary,
        fallback_fails: bool,
        calls: Arc<Mutex<Vec<String>>>,
    }

    impl FakeProvider {
        fn reply(
            &self,
            model: &str,
            tx: Option<&UnboundedSender<String>>,
        ) -> Result<MessagesResponse, MicroClawError> {
            self.calls.lock().unwrap().push(model.to_string());
            if model == "primary" && !matches!(self.primary, Primary::Success) {
                if matches!(self.primary, Primary::FailAfterText) {
                    if let Some(tx) = tx {
                        let _ = tx.send("partial".into());
                    }
                }
                return Err(MicroClawError::LlmApi("primary failure".into()));
            }
            if model == "fallback" && self.fallback_fails {
                return Err(MicroClawError::LlmApi("fallback failure".into()));
            }
            let text = format!("{model}_ok");
            if let Some(tx) = tx {
                let _ = tx.send(text.clone());
            }
            Ok(MessagesResponse {
                content: vec![ResponseContentBlock::Text { text }],
                stop_reason: Some("end_turn".into()),
                usage: None,
            })
        }
    }

    #[async_trait]
    impl LlmProvider for FakeProvider {
        async fn send_message(
            &self,
            _system: &str,
            _messages: Vec<Message>,
            _tools: Option<Vec<ToolDefinition>>,
        ) -> Result<MessagesResponse, MicroClawError> {
            self.reply("primary", None)
        }

        async fn send_message_with_model(
            &self,
            _system: &str,
            _messages: Vec<Message>,
            _tools: Option<Vec<ToolDefinition>>,
            model: Option<&str>,
        ) -> Result<MessagesResponse, MicroClawError> {
            self.reply(model.unwrap_or("primary"), None)
        }

        async fn send_message_stream_with_model(
            &self,
            _system: &str,
            _messages: Vec<Message>,
            _tools: Option<Vec<ToolDefinition>>,
            tx: Option<&UnboundedSender<String>>,
            model: Option<&str>,
        ) -> Result<MessagesResponse, MicroClawError> {
            self.reply(model.unwrap_or("primary"), tx)
        }
    }

    fn fixture(
        primary: Primary,
        fallback_fails: bool,
    ) -> (ResilientProvider, Arc<Mutex<Vec<String>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let inner = Box::new(FakeProvider {
            primary,
            fallback_fails,
            calls: calls.clone(),
        });
        (
            ResilientProvider::new(inner, "primary".into(), "fallback".into()),
            calls,
        )
    }

    fn output(rx: &mut tokio::sync::mpsc::UnboundedReceiver<String>) -> String {
        std::iter::from_fn(|| rx.try_recv().ok()).collect()
    }

    #[tokio::test]
    async fn stream_failure_before_text_falls_back_once() {
        let (provider, calls) = fixture(Primary::FailBeforeText, false);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        provider
            .send_message_stream_with_model("", vec![], None, Some(&tx), Some("primary"))
            .await
            .unwrap();
        assert_eq!(*calls.lock().unwrap(), ["primary", "fallback"]);
        assert_eq!(output(&mut rx), "fallback_ok");
    }

    #[tokio::test]
    async fn stream_failure_after_queued_text_never_retries() {
        let (provider, calls) = fixture(Primary::FailAfterText, false);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        assert!(provider
            .send_message_stream_with_model("", vec![], None, Some(&tx), Some("primary"))
            .await
            .is_err());
        assert_eq!(*calls.lock().unwrap(), ["primary"]);
        assert_eq!(output(&mut rx), "partial");
    }

    #[tokio::test]
    async fn successful_primary_stream_is_forwarded_once() {
        let (provider, calls) = fixture(Primary::Success, false);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        provider
            .send_message_stream_with_model("", vec![], None, Some(&tx), None)
            .await
            .unwrap();
        assert_eq!(*calls.lock().unwrap(), ["primary"]);
        assert_eq!(output(&mut rx), "primary_ok");
    }

    #[tokio::test]
    async fn both_stream_models_fail_without_looping() {
        let (provider, calls) = fixture(Primary::FailBeforeText, true);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let error = provider
            .send_message_stream_with_model("", vec![], None, Some(&tx), None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("primary failure"));
        assert_eq!(*calls.lock().unwrap(), ["primary", "fallback"]);
        assert_eq!(output(&mut rx), "");
    }

    #[tokio::test]
    async fn explicit_fallback_failure_is_not_retried() {
        let (provider, calls) = fixture(Primary::FailBeforeText, true);
        assert!(provider
            .send_message_stream_with_model("", vec![], None, None, Some("fallback"))
            .await
            .is_err());
        assert_eq!(*calls.lock().unwrap(), ["fallback"]);
    }

    #[tokio::test]
    async fn stream_without_text_channel_can_fall_back() {
        let (provider, calls) = fixture(Primary::FailBeforeText, false);
        provider
            .send_message_stream_with_model("", vec![], None, None, Some("primary"))
            .await
            .unwrap();
        assert_eq!(*calls.lock().unwrap(), ["primary", "fallback"]);
    }

    #[tokio::test]
    async fn open_breaker_bypasses_explicit_primary_in_both_call_modes() {
        for streaming in [false, true] {
            let (provider, calls) = fixture(Primary::FailBeforeText, false);
            for _ in 0..BREAKER_FAILURE_THRESHOLD {
                provider.breaker.record_failure(Instant::now());
            }
            if streaming {
                provider
                    .send_message_stream_with_model("", vec![], None, None, Some("primary"))
                    .await
                    .unwrap();
            } else {
                provider
                    .send_message_with_model("", vec![], None, Some("primary"))
                    .await
                    .unwrap();
            }
            assert_eq!(*calls.lock().unwrap(), ["fallback"]);
        }
    }

    #[tokio::test]
    async fn open_breaker_still_honors_other_model_overrides() {
        let (provider, calls) = fixture(Primary::FailBeforeText, false);
        for _ in 0..BREAKER_FAILURE_THRESHOLD {
            provider.breaker.record_failure(Instant::now());
        }
        provider
            .send_message_stream_with_model("", vec![], None, None, Some("other"))
            .await
            .unwrap();
        assert_eq!(*calls.lock().unwrap(), ["other"]);
    }

    #[tokio::test]
    async fn nonstream_failure_still_falls_back() {
        let (provider, calls) = fixture(Primary::FailBeforeText, false);
        provider
            .send_message_with_model("", vec![], None, Some("primary"))
            .await
            .unwrap();
        assert_eq!(*calls.lock().unwrap(), ["primary", "fallback"]);
    }

    #[test]
    pub(crate) fn test_circuit_breaker_opens_and_recovers() {
        let cb = CircuitBreaker::new(2, Duration::from_secs(60));
        let t0 = Instant::now();
        assert!(!cb.is_open(t0));
        cb.record_failure(t0);
        assert!(!cb.is_open(t0), "one failure below threshold must not open");
        cb.record_failure(t0);
        assert!(cb.is_open(t0), "threshold failures must open the breaker");
        assert!(
            cb.is_open(t0 + Duration::from_secs(59)),
            "open during cooldown"
        );
        assert!(
            !cb.is_open(t0 + Duration::from_secs(61)),
            "closed (half-open probe) after cooldown"
        );
        // A success fully resets the breaker.
        cb.record_success();
        assert!(!cb.is_open(t0));
    }
}

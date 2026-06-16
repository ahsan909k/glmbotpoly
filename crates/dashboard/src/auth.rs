//! Token authentication for the dashboard (CLAUDE.md §10/§11).
//!
//! REST requests authenticate with an `Authorization: Bearer <token>` header
//! (the [`require_auth`] middleware). The WebSocket endpoint can't carry a
//! header (browsers forbid it on `WebSocket`), so it authenticates with a
//! `?token=` query param checked in the handler. A configured token is compared
//! in constant time; an unconfigured token is allowed only on a loopback bind
//! (dev convenience), and refused on any other bind (fail closed, mirroring
//! `config::validate`).

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use crate::error::DashboardError;

/// The resolved auth policy for a running server.
#[derive(Debug)]
pub(crate) enum Auth {
    /// A token is configured: every protected request must present it.
    Enforced(Box<[u8]>),
    /// No token AND a loopback bind: allowed unauthenticated (dev).
    OpenLoopback,
}

impl Auth {
    /// Resolves the policy from the bind address and the optional token. A
    /// non-loopback bind without a token is refused (fail closed).
    ///
    /// # Errors
    /// [`DashboardError::NonLoopbackWithoutToken`] when no token is set and the
    /// bind is not loopback.
    pub(crate) fn resolve(bind: SocketAddr, token: Option<&str>) -> Result<Self, DashboardError> {
        match token {
            Some(t) if !t.is_empty() => Ok(Self::Enforced(t.as_bytes().into())),
            _ if bind.ip().is_loopback() => Ok(Self::OpenLoopback),
            _ => Err(DashboardError::NonLoopbackWithoutToken(bind)),
        }
    }

    /// Whether `presented` satisfies the policy.
    pub(crate) fn check(&self, presented: Option<&[u8]>) -> bool {
        match self {
            Self::OpenLoopback => true,
            Self::Enforced(token) => presented.is_some_and(|p| ct_eq(p, token)),
        }
    }
}

/// Length-tolerant constant-time byte comparison: runs over the longer length so
/// a mismatch cannot be timed by early return. A length difference always fails.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff: u8 = u8::from(a.len() != b.len());
    let n = a.len().max(b.len());
    for i in 0..n {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        diff |= x ^ y;
    }
    diff == 0
}

/// The `?token=` query for the WebSocket upgrade.
#[derive(Debug, Deserialize)]
pub(crate) struct WsAuthQuery {
    /// The presented token.
    pub(crate) token: Option<String>,
}

/// JSON error body for a rejected request.
#[derive(Serialize)]
struct ErrBody {
    error: &'static str,
}

/// A `401 Unauthorized` JSON response.
pub(crate) fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(ErrBody {
            error: "unauthorized",
        }),
    )
        .into_response()
}

/// REST auth middleware: rejects with `401` unless the `Authorization: Bearer`
/// token satisfies the policy.
pub(crate) async fn require_auth(
    State(auth): State<Arc<Auth>>,
    req: Request,
    next: Next,
) -> Response {
    let ok = {
        let presented = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "))
            .map(str::as_bytes);
        auth.check(presented)
    };
    if ok {
        next.run(req).await
    } else {
        unauthorized()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};

    use super::*;

    fn loopback() -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8080)
    }
    fn public() -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 8080)
    }

    #[test]
    fn ct_eq_matches_only_equal() {
        assert!(ct_eq(b"secret", b"secret"));
        assert!(!ct_eq(b"secret", b"secreT"));
        assert!(!ct_eq(b"secret", b"secret-longer"));
        assert!(!ct_eq(b"", b"x"));
        assert!(ct_eq(b"", b""));
    }

    #[test]
    fn resolve_matrix() {
        // Some token → Enforced regardless of bind.
        assert!(matches!(
            Auth::resolve(public(), Some("tok")),
            Ok(Auth::Enforced(_))
        ));
        // None + loopback → OpenLoopback.
        assert!(matches!(
            Auth::resolve(loopback(), None),
            Ok(Auth::OpenLoopback)
        ));
        // Empty token treated as absent.
        assert!(matches!(
            Auth::resolve(loopback(), Some("")),
            Ok(Auth::OpenLoopback)
        ));
        // None + non-loopback → refuse.
        assert!(matches!(
            Auth::resolve(public(), None),
            Err(DashboardError::NonLoopbackWithoutToken(_))
        ));
    }

    #[test]
    fn check_enforces_or_opens() {
        let enforced = Auth::Enforced(b"secret".as_slice().into());
        assert!(enforced.check(Some(b"secret")));
        assert!(!enforced.check(Some(b"wrong")));
        assert!(!enforced.check(None));

        let open = Auth::OpenLoopback;
        assert!(open.check(None));
        assert!(open.check(Some(b"anything")));
    }
}

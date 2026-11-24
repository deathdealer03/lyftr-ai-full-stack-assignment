import React, { useState, useEffect, useRef } from 'react';
import {
  Send,
  Zap,
  Database,
  Cpu,
  Clock,
  Coins,
  ShieldAlert,
  AlertCircle,
  RefreshCw,
  Sliders,
  CheckCircle2,
  Terminal,
  Sparkles,
  Layers,
  ArrowRight,
  Info
} from 'lucide-react';
import { ChatMessage, HealthStatus } from './types';
import { fetchHealth, sendChatMessage, API_BASE_URL } from './api';

// Generate client-side session ID on initial load: s-<random 6-8 chars>
function generateSessionId(): string {
  const rand = Math.random().toString(36).substring(2, 8);
  return `s-${rand}`;
}

export const App: React.FC = () => {
  const [sessionId, setSessionId] = useState<string>(() => generateSessionId());
  const [systemPrompt, setSystemPrompt] = useState<string>('');
  const [inputMessage, setInputMessage] = useState<string>('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  // Failure simulation hook control for demo and testing
  const [simulateFailure, setSimulateFailure] = useState<string>('');
  const [showConfig, setShowConfig] = useState<boolean>(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom of conversation
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  // Initial health check
  const checkBackendHealth = async () => {
    try {
      const data = await fetchHealth();
      setHealth(data);
      setHealthError(null);
    } catch (err: any) {
      setHealthError(err.message || 'Backend unreachable');
      setHealth(null);
    }
  };

  useEffect(() => {
    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Reset to brand new session
  const handleNewSession = () => {
    const newId = generateSessionId();
    setSessionId(newId);
    setMessages([]);
    setInputMessage('');
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  // Send message
  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = inputMessage.trim();
    if (!trimmed || loading) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: trimmed,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    // Add user message to view
    setMessages((prev) => [...prev, userMessage]);
    // Clear input immediately as required by specification
    setInputMessage('');
    setLoading(true);

    try {
      const response = await sendChatMessage(
        sessionId,
        trimmed,
        systemPrompt || undefined,
        simulateFailure || undefined
      );

      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: response.reply,
        metadata: response.metadata,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
      const errorMessageText = err.message || 'An unexpected error occurred';
      const errorItem: ChatMessage = {
        id: `error-${Date.now()}`,
        role: 'error',
        content: errorMessageText,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorItem]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', maxWidth: '1180px', margin: '0 auto', padding: '12px 16px' }}>
      {/* Top Header */}
      <header style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '12px 18px',
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: '12px',
        border: '1px solid var(--border-subtle)',
        marginBottom: '12px',
        boxShadow: 'var(--shadow-sm)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '38px',
            height: '38px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 2px 8px rgba(59, 130, 246, 0.3)'
          }}>
            <Sparkles size={20} color="#fff" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.15rem', fontWeight: 700, letterSpacing: '-0.02em', color: '#fff' }}>
              Lyftr AI <span style={{ fontSize: '0.8rem', fontWeight: 500, color: 'var(--text-muted)' }}>Gateway</span>
            </h1>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '-2px' }}>
              Key Failover &middot; In-Memory/Disk Caching &middot; SQLite Multi-Turn
            </p>
          </div>
        </div>

        {/* Health status & Quick Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {health ? (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 10px',
              borderRadius: '20px',
              backgroundColor: 'var(--accent-green-subtle)',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              fontSize: '0.75rem',
              color: '#34d399',
              fontWeight: 500
            }}>
              <CheckCircle2 size={13} />
              <span>Healthy ({health.keys_configured} keys &bull; Cache On)</span>
            </div>
          ) : (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 10px',
              borderRadius: '20px',
              backgroundColor: 'var(--accent-rose-subtle)',
              border: '1px solid rgba(244, 63, 94, 0.3)',
              fontSize: '0.75rem',
              color: '#fb7185',
              fontWeight: 500
            }}>
              <AlertCircle size={13} />
              <span>{healthError || 'Connecting...'}</span>
            </div>
          )}

          {/* Session Pill */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '4px 10px',
            borderRadius: '8px',
            backgroundColor: 'var(--bg-surface)',
            border: '1px solid var(--border-subtle)',
            fontSize: '0.75rem',
            fontFamily: 'var(--font-mono)',
            color: 'var(--text-muted)'
          }}>
            <span>Session:</span>
            <strong style={{ color: '#60a5fa' }}>{sessionId}</strong>
          </div>

          <button
            onClick={handleNewSession}
            title="Start new fresh session"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 12px',
              borderRadius: '8px',
              backgroundColor: 'var(--bg-surface)',
              border: '1px solid var(--border-subtle)',
              color: 'var(--text-main)',
              fontSize: '0.75rem',
              fontWeight: 500
            }}
            onMouseOver={(e) => (e.currentTarget.style.backgroundColor = 'var(--bg-surface-hover)')}
            onMouseOut={(e) => (e.currentTarget.style.backgroundColor = 'var(--bg-surface)')}
          >
            <RefreshCw size={13} />
            <span>New Session</span>
          </button>

          <button
            onClick={() => setShowConfig(!showConfig)}
            title="Toggle System Prompt & Simulation Panel"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 12px',
              borderRadius: '8px',
              backgroundColor: showConfig ? 'var(--accent-blue-subtle)' : 'var(--bg-surface)',
              border: showConfig ? '1px solid var(--accent-blue)' : '1px solid var(--border-subtle)',
              color: showConfig ? '#93c5fd' : 'var(--text-main)',
              fontSize: '0.75rem',
              fontWeight: 500
            }}
          >
            <Sliders size={13} />
            <span>Controls</span>
          </button>
        </div>
      </header>

      {/* Expandable Controls Panel (System Prompt & Failure Simulator) */}
      {showConfig && (
        <div style={{
          backgroundColor: 'var(--bg-secondary)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '12px',
          padding: '14px 18px',
          marginBottom: '12px',
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '16px',
          boxShadow: 'var(--shadow-md)'
        }}>
          {/* System Prompt Input */}
          <div>
            <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '6px' }}>
              Session System Prompt (Optional)
            </label>
            <input
              data-testid="system-prompt-input"
              type="text"
              placeholder="e.g. You are a concise mathematical assistant."
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              style={{
                width: '100%',
                padding: '8px 12px',
                borderRadius: '8px',
                fontSize: '0.82rem',
              }}
            />
            <p style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: '4px' }}>
              Overrides the session prompt. Never included in the history endpoint.
            </p>
          </div>

          {/* Failover Test Hook (X-Simulate-Failure) */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
              <label style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                Failover Test Hook (X-Simulate-Failure)
              </label>
              {simulateFailure && (
                <button
                  onClick={() => setSimulateFailure('')}
                  style={{ background: 'none', color: '#f87171', fontSize: '0.7rem', textDecoration: 'underline' }}
                >
                  Clear Hook
                </button>
              )}
            </div>
            <div style={{ display: 'flex', gap: '6px', marginBottom: '6px' }}>
              {[
                { label: 'None', val: '' },
                { label: 'key_1=429', val: 'key_1=429' },
                { label: 'key_1=500', val: 'key_1=500' },
                { label: '*=429 (503)', val: '*=429' },
              ].map((btn) => (
                <button
                  key={btn.val}
                  type="button"
                  onClick={() => setSimulateFailure(btn.val)}
                  style={{
                    padding: '4px 10px',
                    borderRadius: '6px',
                    fontSize: '0.72rem',
                    fontFamily: 'var(--font-mono)',
                    backgroundColor: simulateFailure === btn.val ? 'var(--accent-blue)' : 'var(--bg-surface)',
                    color: simulateFailure === btn.val ? '#fff' : 'var(--text-muted)',
                    border: '1px solid var(--border-subtle)',
                  }}
                >
                  {btn.label}
                </button>
              ))}
            </div>
            <input
              type="text"
              placeholder="Custom pair, e.g. key_1=429,key_2=500"
              value={simulateFailure}
              onChange={(e) => setSimulateFailure(e.target.value)}
              style={{
                width: '100%',
                padding: '6px 10px',
                borderRadius: '6px',
                fontSize: '0.75rem',
                fontFamily: 'var(--font-mono)',
              }}
            />
          </div>
        </div>
      )}

      {/* Main Chat Stream Container */}
      <main style={{
        flex: 1,
        overflowY: 'auto',
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border-subtle)',
        borderRadius: '12px',
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '18px',
        boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.1)'
      }}>
        {messages.length === 0 ? (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            color: 'var(--text-dim)',
            textAlign: 'center',
            padding: '20px'
          }}>
            <div style={{
              width: '56px',
              height: '56px',
              borderRadius: '16px',
              backgroundColor: 'var(--bg-surface)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '16px',
              border: '1px solid var(--border-subtle)'
            }}>
              <Terminal size={26} color="#60a5fa" />
            </div>
            <h2 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '6px' }}>
              Ready for Conversation
            </h2>
            <p style={{ fontSize: '0.85rem', maxWidth: '440px', lineHeight: 1.5 }}>
              Ask a question to see <strong>LIVE_INFERENCE</strong>. Repeat the exact query in another session to verify <strong>LLM_CACHE</strong>. Use the controls to test key failover.
            </p>
            <div style={{ display: 'flex', gap: '8px', marginTop: '18px', flexWrap: 'wrap', justifyContent: 'center' }}>
              {[
                'What is the capital of France?',
                'Explain quantum computing in one sentence.',
                'My name is Alex. Remember this.'
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => setInputMessage(suggestion)}
                  style={{
                    padding: '6px 12px',
                    borderRadius: '20px',
                    backgroundColor: 'var(--bg-surface)',
                    border: '1px solid var(--border-subtle)',
                    color: 'var(--text-muted)',
                    fontSize: '0.78rem'
                  }}
                  onMouseOver={(e) => (e.currentTarget.style.borderColor = 'var(--accent-blue)')}
                  onMouseOut={(e) => (e.currentTarget.style.borderColor = 'var(--border-subtle)')}
                >
                  "{suggestion}"
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start',
                width: '100%'
              }}
            >
              {/* User Message Bubble */}
              {msg.role === 'user' && (
                <div style={{
                  maxWidth: '75%',
                  backgroundColor: 'var(--accent-blue)',
                  color: '#ffffff',
                  padding: '12px 16px',
                  borderRadius: '16px 16px 2px 16px',
                  fontSize: '0.92rem',
                  lineHeight: 1.5,
                  wordBreak: 'break-word',
                  boxShadow: '0 2px 8px rgba(59, 130, 246, 0.25)'
                }}>
                  {msg.content}
                </div>
              )}

              {/* Error Message Bubble */}
              {msg.role === 'error' && (
                <div style={{
                  maxWidth: '85%',
                  backgroundColor: 'var(--accent-rose-subtle)',
                  border: '1px solid rgba(244, 63, 94, 0.35)',
                  color: '#fecdd3',
                  padding: '14px 18px',
                  borderRadius: '14px',
                  fontSize: '0.88rem',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '10px'
                }}>
                  <ShieldAlert size={20} color="#f43f5e" style={{ flexShrink: 0, marginTop: '2px' }} />
                  <div>
                    <strong style={{ color: '#fda4af', display: 'block', marginBottom: '2px' }}>
                      Request Error
                    </strong>
                    <span>{msg.content}</span>
                  </div>
                </div>
              )}

              {/* Assistant Message Container (Exact data-testid="assistant-message") */}
              {msg.role === 'assistant' && (
                <div
                  data-testid="assistant-message"
                  style={{
                    maxWidth: '85%',
                    backgroundColor: 'var(--bg-surface)',
                    border: '1px solid var(--border-subtle)',
                    padding: '16px 20px',
                    borderRadius: '16px 16px 16px 2px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '14px',
                    boxShadow: 'var(--shadow-sm)'
                  }}
                >
                  {/* Reply Text */}
                  <div style={{
                    fontSize: '0.93rem',
                    lineHeight: 1.6,
                    color: 'var(--text-main)',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word'
                  }}>
                    {msg.content}
                  </div>

                  {/* Metadata Bar required by contract */}
                  {msg.metadata && (
                    <div style={{
                      display: 'flex',
                      flexWrap: 'wrap',
                      alignItems: 'center',
                      gap: '8px',
                      paddingTop: '10px',
                      borderTop: '1px solid var(--border-subtle)',
                      fontSize: '0.74rem'
                    }}>
                      {/* Exact meta-source */}
                      <span
                        data-testid="meta-source"
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '5px',
                          padding: '3px 9px',
                          borderRadius: '6px',
                          fontWeight: 600,
                          backgroundColor: msg.metadata.resolution_source === 'LLM_CACHE'
                            ? 'var(--accent-green-subtle)'
                            : 'var(--accent-purple-subtle)',
                          color: msg.metadata.resolution_source === 'LLM_CACHE'
                            ? '#34d399'
                            : '#c084fc',
                          border: `1px solid ${
                            msg.metadata.resolution_source === 'LLM_CACHE'
                              ? 'rgba(16, 185, 129, 0.3)'
                              : 'rgba(139, 92, 246, 0.3)'
                          }`
                        }}
                      >
                        {msg.metadata.resolution_source === 'LLM_CACHE' ? (
                          <Database size={12} />
                        ) : (
                          <Zap size={12} />
                        )}
                        {msg.metadata.resolution_source}
                      </span>

                      {/* Exact meta-model */}
                      <span
                        data-testid="meta-model"
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '4px',
                          padding: '3px 8px',
                          borderRadius: '6px',
                          backgroundColor: 'var(--bg-secondary)',
                          color: 'var(--text-muted)',
                          fontFamily: 'var(--font-mono)'
                        }}
                      >
                        <Cpu size={12} />
                        {msg.metadata.model}
                      </span>

                      {/* Exact meta-latency */}
                      <span
                        data-testid="meta-latency"
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '4px',
                          padding: '3px 8px',
                          borderRadius: '6px',
                          backgroundColor: 'var(--bg-secondary)',
                          color: '#fbbf24',
                          fontWeight: 500
                        }}
                      >
                        <Clock size={12} />
                        {msg.metadata.latency_ms} ms
                      </span>

                      {/* Exact meta-input-tokens */}
                      <span
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '4px',
                          padding: '3px 8px',
                          borderRadius: '6px',
                          backgroundColor: 'var(--bg-secondary)',
                          color: 'var(--text-muted)'
                        }}
                      >
                        <span>In:</span>
                        <strong data-testid="meta-input-tokens" style={{ color: '#93c5fd' }}>
                          {msg.metadata.input_tokens}
                        </strong>
                      </span>

                      {/* Exact meta-output-tokens */}
                      <span
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '4px',
                          padding: '3px 8px',
                          borderRadius: '6px',
                          backgroundColor: 'var(--bg-secondary)',
                          color: 'var(--text-muted)'
                        }}
                      >
                        <span>Out:</span>
                        <strong data-testid="meta-output-tokens" style={{ color: '#93c5fd' }}>
                          {msg.metadata.output_tokens}
                        </strong>
                      </span>

                      {/* Key ID badge */}
                      {msg.metadata.key_id && (
                        <span style={{
                          padding: '3px 8px',
                          borderRadius: '6px',
                          backgroundColor: 'rgba(59, 130, 246, 0.1)',
                          color: '#60a5fa',
                          fontFamily: 'var(--font-mono)'
                        }}>
                          key: {msg.metadata.key_id}
                        </span>
                      )}

                      {/* Fallback attempts badge */}
                      {msg.metadata.fallback_attempts > 0 && (
                        <span style={{
                          padding: '3px 8px',
                          borderRadius: '6px',
                          backgroundColor: 'var(--accent-amber-subtle)',
                          color: '#fbbf24',
                          fontWeight: 600
                        }}>
                          Failovers: {msg.metadata.fallback_attempts}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))
        )}

        {/* Loading Indicator */}
        {loading && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            padding: '12px 18px',
            backgroundColor: 'var(--bg-surface)',
            borderRadius: '12px',
            width: 'fit-content',
            border: '1px solid var(--border-subtle)'
          }}>
            <div style={{
              width: '16px',
              height: '16px',
              border: '2px solid rgba(59, 130, 246, 0.3)',
              borderTopColor: 'var(--accent-blue)',
              borderRadius: '50%',
              animation: 'spin 0.8s linear infinite'
            }} />
            <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              Generating response with key failover & caching...
            </span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </main>

      {/* Bottom Input Area */}
      <footer style={{ marginTop: '12px' }}>
        <form
          onSubmit={handleSendMessage}
          style={{
            display: 'flex',
            alignItems: 'flex-end',
            gap: '10px',
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '14px',
            padding: '10px 14px',
            boxShadow: 'var(--shadow-md)'
          }}
        >
          <textarea
            ref={textareaRef}
            data-testid="chat-input"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message... (Enter to send, Shift+Enter for newline)"
            rows={1}
            disabled={loading}
            style={{
              flex: 1,
              backgroundColor: 'transparent',
              border: 'none',
              resize: 'none',
              outline: 'none',
              padding: '6px 4px',
              fontSize: '0.92rem',
              color: 'var(--text-main)',
              maxHeight: '120px',
              lineHeight: 1.4
            }}
          />

          <button
            data-testid="send-button"
            type="submit"
            disabled={loading || !inputMessage.trim()}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '38px',
              height: '38px',
              borderRadius: '10px',
              backgroundColor: inputMessage.trim() && !loading ? 'var(--accent-blue)' : 'var(--bg-surface)',
              color: inputMessage.trim() && !loading ? '#fff' : 'var(--text-dim)',
              border: '1px solid var(--border-subtle)',
              cursor: inputMessage.trim() && !loading ? 'pointer' : 'not-allowed',
              flexShrink: 0
            }}
            title="Send Message"
          >
            <Send size={16} />
          </button>
        </form>

        {/* Footer info strip */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '6px 8px 0 8px',
          fontSize: '0.7rem',
          color: 'var(--text-dim)'
        }}>
          <span>Backend Target: <code style={{ color: 'var(--text-muted)' }}>{API_BASE_URL}</code></span>
          <span>FastAPI &middot; LiteLLM Key Failover &middot; SQLite Session Cache</span>
        </div>
      </footer>

      {/* Global CSS animation for spinner */}
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  BookOpen,
  Send,
  Sparkles,
  Sliders,
  RefreshCw,
  Trash2,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  Cpu,
  Database,
  Search,
  Book,
  Menu,
  X,
  ExternalLink,
  HelpCircle,
  Clock
} from 'lucide-react';
import './App.css';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [inputQuery, setInputQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const [suggestedQuestions, setSuggestedQuestions] = useState([]);
  const [topK, setTopK] = useState(6);
  const [temperature, setTemperature] = useState(0.0);
  const [expandedCitations, setExpandedCitations] = useState({});
  const [copiedId, setCopiedId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll to latest message
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Fetch initial status and suggested prompts
  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setSystemStatus(data);
      }
    } catch (err) {
      console.error('Failed to load system status:', err);
    }
  };

  const fetchSuggested = async () => {
    try {
      const res = await fetch('/api/suggested');
      if (res.ok) {
        const data = await res.json();
        setSuggestedQuestions(data.questions || []);
      }
    } catch (err) {
      console.error('Failed to load suggestions:', err);
    }
  };

  useEffect(() => {
    fetchStatus();
    fetchSuggested();
  }, []);

  // Textarea auto-resize
  const handleTextareaChange = (e) => {
    setInputQuery(e.target.value);
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 140)}px`;
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = async (queryOverride = null) => {
    const query = (queryOverride || inputQuery).trim();
    if (!query || isLoading) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: query,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputQuery('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
    setIsLoading(true);

    // Format conversational history for Flask RAG API
    const historyPayload = messages.slice(-6).map((m) => ({
      role: m.role,
      content: m.content,
    }));

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: query,
          history: historyPayload,
          top_k: topK,
          temperature: temperature,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Server error occurred during retrieval');
      }

      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.answer,
        citations: data.citations || [],
        search_queries: data.search_queries || [],
        latency_ms: data.latency_ms,
        model: data.model,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `⚠️ **Error:** ${err.message}\n\nPlease verify that your Flask backend is running and that your API keys are configured in \`.env\`.`,
        isError: true,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleCitations = (messageId) => {
    setExpandedCitations((prev) => ({
      ...prev,
      [messageId]: !prev[messageId],
    }));
  };

  const handleCopy = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const clearChat = () => {
    setMessages([]);
    setExpandedCitations({});
  };

  return (
    <div className="app-container">
      {/* ====================================================================
          Left Sidebar: Knowledge Base & RAG Controls
          ==================================================================== */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="logo-icon-wrapper">
            <BookOpen size={22} />
          </div>
          <div>
            <div className="brand-title">Study-RAG</div>
            <div className="brand-tag">
              <Sparkles size={11} /> AI Academic Companion
            </div>
          </div>
        </div>

        {/* System Status Card */}
        <div className="status-card">
          <div className="status-header">
            <span className="section-label" style={{ marginBottom: 0 }}>
              <Database size={13} /> Vector Store
            </span>
            <span className={`status-badge ${systemStatus?.database_ready ? '' : 'warning'}`}>
              <span className="pulse-dot"></span>
              {systemStatus?.database_ready ? 'Connected' : 'Offline'}
            </span>
          </div>

          <div className="status-meta-item">
            <span>Indexed Chunks</span>
            <span className="meta-value">
              {systemStatus?.chunk_count ? systemStatus.chunk_count.toLocaleString() : '0'}
            </span>
          </div>
          <div className="status-meta-item">
            <span>Embeddings</span>
            <span className="meta-value">all-MiniLM-L6-v2</span>
          </div>
          <div className="status-meta-item">
            <span>LLM Provider</span>
            <span className="meta-value">{systemStatus?.provider || 'Groq'}</span>
          </div>
          <div className="status-meta-item">
            <span>Active Model</span>
            <span className="meta-value" style={{ fontSize: '0.72rem' }}>
              {systemStatus?.model || 'gpt-oss-120b'}
            </span>
          </div>
        </div>

        {/* Textbooks Shelf */}
        <div className="sidebar-section">
          <div className="section-label">
            <Book size={13} /> Active Curriculum
          </div>
          {systemStatus?.textbooks && systemStatus.textbooks.length > 0 ? (
            systemStatus.textbooks.map((book, idx) => (
              <div key={idx} className="textbook-card">
                <BookOpen size={16} className="textbook-icon" />
                <span className="textbook-title" title={book}>
                  {book.replace('.pdf', '')} (Mankiw Economics)
                </span>
              </div>
            ))
          ) : (
            <div className="textbook-card">
              <BookOpen size={16} className="textbook-icon" />
              <span className="textbook-title">Economics 10th Ed.</span>
            </div>
          )}
        </div>

        {/* Retrieval Controls */}
        <div className="sidebar-section">
          <div className="section-label">
            <Sliders size={13} /> Tuning Parameters
          </div>
          <div className="slider-group">
            <div className="slider-item">
              <div className="slider-label-row">
                <span>Retrieved Context (k)</span>
                <span className="slider-val">{topK} chunks</span>
              </div>
              <input
                type="range"
                min="2"
                max="12"
                step="1"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="range-input"
              />
            </div>

            <div className="slider-item">
              <div className="slider-label-row">
                <span>Temperature</span>
                <span className="slider-val">{temperature.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="range-input"
              />
            </div>
          </div>
        </div>

        {/* Sidebar Actions */}
        <div className="sidebar-footer">
          <button onClick={clearChat} className="btn-sidebar danger" title="Clear conversation">
            <Trash2 size={15} /> Clear Chat
          </button>
          <button onClick={fetchStatus} className="btn-sidebar" title="Refresh database state">
            <RefreshCw size={15} /> Sync Vector DB
          </button>
        </div>
      </aside>

      {/* ====================================================================
          Main Chat Workspace
          ==================================================================== */}
      <main className="chat-workspace">
        {/* Top Navbar */}
        <header className="chat-header">
          <div className="header-title-area">
            <button
              className="mobile-menu-btn"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              aria-label="Toggle menu"
            >
              {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
            <div>
              <div className="chat-title">Textbook Q&A Assistant</div>
              <div className="chat-subtitle">Grounded in verified page citations</div>
            </div>
          </div>

          <div className="header-badges">
            <div className="chip-tag">
              <Database size={12} color="var(--accent-emerald)" />
              {systemStatus?.chunk_count ? `${systemStatus.chunk_count.toLocaleString()} Chunks` : 'Online'}
            </div>
            <div className="chip-tag">
              <Cpu size={12} color="var(--primary-light)" />
              Zero-Cost ONNX
            </div>
          </div>
        </header>

        {/* Messages Feed */}
        <div className="message-feed">
          {messages.length === 0 ? (
            <div className="empty-state animate-slide-up">
              <div className="hero-icon-bubble">
                <BookOpen size={36} />
              </div>
              <h1 className="empty-headline">What are you studying today?</h1>
              <p className="empty-desc">
                Ask any question from your economics textbook. Our adaptive tolerance chunker
                retrieves exact source passages and generates answers with guaranteed page citations.
              </p>

              <div className="suggestions-grid">
                {suggestedQuestions.map((q, idx) => (
                  <button
                    key={idx}
                    className="suggestion-btn"
                    onClick={() => handleSend(q)}
                  >
                    <Search size={16} className="suggestion-icon" />
                    <span>{q}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div
                key={msg.id}
                className={`message-row ${msg.role === 'user' ? 'user' : 'assistant'} animate-slide-up`}
              >
                <div className={`avatar ${msg.role === 'user' ? 'user-avatar' : 'ai-avatar'}`}>
                  {msg.role === 'user' ? 'U' : <Sparkles size={18} />}
                </div>

                <div className="message-body">
                  {msg.role === 'user' ? (
                    <div className="user-bubble">{msg.content}</div>
                  ) : (
                    <div className="ai-bubble">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>

                      {/* Search Queries Breakdown */}
                      {msg.search_queries && msg.search_queries.length > 0 && (
                        <div className="query-breakdown">
                          <span>🔍 Vector Searches:</span>
                          {msg.search_queries.map((q, i) => (
                            <span key={i} className="query-chip">
                              "{q}"
                            </span>
                          ))}
                        </div>
                      )}

                      {/* Collapsible Verified Citations Accordion */}
                      {msg.citations && msg.citations.length > 0 && (
                        <div className="citations-box">
                          <button
                            className="citations-toggle"
                            onClick={() => toggleCitations(msg.id)}
                          >
                            <span>
                              📖 Verified Textbook Citations ({msg.citations.length} excerpts)
                            </span>
                            {expandedCitations[msg.id] ? (
                              <ChevronUp size={16} />
                            ) : (
                              <ChevronDown size={16} />
                            )}
                          </button>

                          {expandedCitations[msg.id] && (
                            <div className="citation-list animate-slide-up">
                              {msg.citations.map((cit, cIdx) => (
                                <div key={cIdx} className="citation-card">
                                  <div className="citation-card-header">
                                    <span className="citation-page-badge">
                                      Page {cit.page}
                                    </span>
                                    <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>
                                      {cit.source}
                                    </span>
                                  </div>
                                  <p className="citation-text">"{cit.text}"</p>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Metadata and Quick Actions */}
                  {msg.role === 'assistant' && !msg.isError && (
                    <div className="ai-actions-bar">
                      <button
                        className="btn-action-copy"
                        onClick={() => handleCopy(msg.content, msg.id)}
                        title="Copy answer"
                      >
                        {copiedId === msg.id ? (
                          <>
                            <Check size={13} color="var(--accent-emerald)" />
                            <span>Copied!</span>
                          </>
                        ) : (
                          <>
                            <Copy size={13} />
                            <span>Copy answer</span>
                          </>
                        )}
                      </button>
                      {msg.latency_ms && (
                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <Clock size={12} /> {msg.latency_ms}ms
                        </span>
                      )}
                      <span>• {msg.timestamp}</span>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          {/* Thinking / Searching Indicator */}
          {isLoading && (
            <div className="message-row assistant animate-slide-up">
              <div className="avatar ai-avatar">
                <Sparkles size={18} />
              </div>
              <div className="message-body">
                <div className="thinking-bubble">
                  <div className="spinner-dots">
                    <span className="spinner-dot"></span>
                    <span className="spinner-dot"></span>
                    <span className="spinner-dot"></span>
                  </div>
                  <span>Searching textbook vectors and formulating response...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className="chat-input-wrapper">
          <div className="input-container">
            <textarea
              ref={textareaRef}
              rows={1}
              value={inputQuery}
              onChange={handleTextareaChange}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your textbook (e.g. 'What is the 7th principle of economics?')..."
              className="chat-textarea"
              disabled={isLoading}
            />
            <button
              onClick={() => handleSend()}
              disabled={!inputQuery.trim() || isLoading}
              className="send-btn"
              title="Send question"
            >
              <Send size={18} />
            </button>
          </div>
          <div className="input-hints">
            <span>💡 <strong>Tip:</strong> You can ask by chapter, principle number, or concept name.</span>
            <span>Press <strong>Enter</strong> to send • <strong>Shift+Enter</strong> for newline</span>
          </div>
        </div>
      </main>
    </div>
  );
}

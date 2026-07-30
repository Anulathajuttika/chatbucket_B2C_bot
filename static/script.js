// --- Session handling ---
// Session ids are now issued by the server (unguessable, random) rather than
// generated client-side, so a session id can't be forged/guessed by another
// visitor to read someone else's conversation.
localStorage.clear()
let sessionId = localStorage.getItem("chatbucket_session_id");

console.log("====================================");
console.log("Initial session from localStorage:");
console.log(sessionId);
console.log("====================================");

async function ensureSessionId() {
  console.log("ensureSessionId() called");

  if (sessionId) {
    console.log("Existing session found:");
    console.log(sessionId);
    return sessionId;
  }

  console.log("No session found. Requesting /session/new...");

  try {
    const res = await fetch("/session/new");
    const data = await res.json();

    console.log("/session/new response:");
    console.log(data);

    sessionId = data.session_id;

    console.log("New session received:");
    console.log(sessionId);

    localStorage.setItem("chatbucket_session_id", sessionId);

    console.log("Saved session to localStorage.");
  } catch (err) {
    console.error("Failed to obtain session id:");
    console.error(err);
  }

  return sessionId;
}

async function newSessionId() {
  console.log("Requesting a completely new session...");

  try {
    const res = await fetch("/session/new");
    const data = await res.json();

    console.log("New session response:");
    console.log(data);

    return data.session_id;
  } catch (err) {
    console.error("Failed to obtain new session id:");
    console.error(err);

    return sessionId;
  }
}

const chatWindow = document.getElementById("chatWindow");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const MAX_MESSAGE_LENGTH = 2000; // keep in sync with config.MAX_MESSAGE_LENGTH on the server

// --- Auto-resize textarea ---
userInput.addEventListener("input", () => {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 160) + "px";
});

// --- Enter to send (Shift+Enter for newline) ---
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener("click", sendMessage);

// --- New chat ---
newChatBtn.addEventListener("click", async () => {
  sessionId = await newSessionId();
  localStorage.setItem("chatbucket_session_id", sessionId);
  chatWindow.innerHTML = `
    <div class="message bot-message">
      <div class="avatar bot-avatar">CB</div>
      <div class="bubble">
        Hi! I'm the ChatBucket assistant. Ask me anything about
        <strong>ChatBucket</strong> or <strong>ToDoZee</strong>.
      </div>
    </div>`;
});

function appendMessage(role, htmlContent) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role === "user" ? "user-message" : "bot-message"}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role === "user" ? "user-avatar" : "bot-avatar"}`;
  avatar.textContent = role === "user" ? "You" : "CB";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = htmlContent;

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  chatWindow.appendChild(wrapper);
  scrollToBottom();
  return bubble;
}

function appendTypingIndicator() {
  const wrapper = document.createElement("div");
  wrapper.className = "message bot-message";
  wrapper.id = "typingIndicator";

  wrapper.innerHTML = `
    <div class="avatar bot-avatar">CB</div>
    <div class="bubble">
      <div class="typing-dots"><span></span><span></span><span></span></div>
    </div>`;

  chatWindow.appendChild(wrapper);
  scrollToBottom();
}

function removeTypingIndicator() {
  const el = document.getElementById("typingIndicator");
  if (el) el.remove();
}

function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  if (text.length > MAX_MESSAGE_LENGTH) {
    appendMessage("bot", `That message is too long (max ${MAX_MESSAGE_LENGTH} characters). Please shorten it.`);
    return;
  }

  const sid = await ensureSessionId();
  if (!sid) {
    appendMessage(
      "bot",
      "Couldn't start a session — please refresh and try again."
    );
    return;
  }
  
  console.log("====================================");
  console.log("Preparing Chat Request");
  console.log("Session ID:");
  console.log(sid);
  
  console.log("User Message:");
  console.log(text);
  
  const payload = {
    session_id: sid,
    message: text,
  };
  
  console.log("Payload:");
  console.log(payload);
  
  console.log("Payload JSON:");
  console.log(JSON.stringify(payload));
  
  console.log("====================================");
  
  appendMessage("user", escapeHtml(text));
  userInput.value = "";
  userInput.style.height = "auto";
  sendBtn.disabled = true;
  

  appendTypingIndicator();

  try {
    console.log("Session ID:", sid);
    console.log("Request Body:") 
    const payload = {
      session_id: sid,
      message: text,
    };
    
    console.log("Sending POST /chat");
    console.log(payload);
    
    const res = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    
    console.log("Response Status:");
    console.log(res.status);
    
    console.log("Response OK:");
    console.log(res.ok);

    removeTypingIndicator();

    if (res.status === 429) {
      appendMessage("bot", "You're sending messages a bit fast — please wait a moment and try again.");
      return;
    }
    if (res.status >= 400) {
      appendMessage("bot", "Sorry, that request couldn't be processed. Please try again.");
      return;
    }

    const data = await res.json();
    console.log("Response Body:");
    console.log(data);
    const rawHtml = window.marked ? marked.parse(data.response || "") : (data.response || "");
    // Sanitize before inserting as HTML — the bot response passes through an
    // LLM, and markdown rendering allows raw HTML through by default, so
    // this is required to prevent stored/reflected XSS.
    const safeHtml = window.DOMPurify ? DOMPurify.sanitize(rawHtml) : escapeHtml(data.response || "");
    appendMessage("bot", safeHtml);
  } catch (err) {
    removeTypingIndicator();
    appendMessage("bot", "Sorry, I couldn't reach the server. Please try again.");
    console.error(err);
  } finally {
    sendBtn.disabled = false;
    userInput.focus();
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
} 
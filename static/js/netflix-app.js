// Netflix-style app functionality
let selectedMovie = null;
let selectedGenre = null;
let adTimer = null;
let skipCountdown = 5;
let currentAd = null;

// Play movie and show ad
function playMovie(movieId, genre) {
    selectedMovie = movieId;
    selectedGenre = genre;
    
    // Get user profile
    const formData = new FormData(document.getElementById('userProfileForm'));
    const userProfile = {};
    for (let [key, value] of formData.entries()) {
        userProfile[key] = isNaN(value) ? value : parseInt(value);
    }
    
    // Request optimized ad
    fetch('/api/optimize_ad', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            user_profile: userProfile,
            content_context: { movie_id: movieId, genre: genre }
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentAd = data.ad;
            showVideoPlayer();
            // Show ad after 10 seconds of video
            setTimeout(() => showAd(data.ad), 10000);
        }
    });
}

// Show video player
function showVideoPlayer() {
    document.getElementById('videoPlayerModal').style.display = 'flex';
    const video = document.getElementById('mainVideo');
    video.play();
}

// Show ad overlay
function showAd(adData) {
    const adOverlay = document.getElementById('adOverlay');
    const adContent = document.getElementById('adContent');
    
    // Pause video
    document.getElementById('mainVideo').pause();
    
    // Set ad content
    adContent.innerHTML = `
        <h2>${adData.title}</h2>
        <p>${adData.personalized_copy || 'This is a demo ad for testing purposes'}</p>
        <div class="ad-stats">
            <small>Predicted CTR: ${(adData.predicted_ctr * 100).toFixed(2)}%</small>
        </div>
    `;
    
    // Show overlay
    adOverlay.style.display = 'flex';
    
    // Start countdown
    startSkipCountdown();
}

// Skip countdown
function startSkipCountdown() {
    skipCountdown = 5;
    const countdownEl = document.getElementById('countdown');
    const skipButton = document.getElementById('skipButton');
    const skipTimer = document.getElementById('skipTimer');
    
    adTimer = setInterval(() => {
        skipCountdown--;
        countdownEl.textContent = skipCountdown;
        
        if (skipCountdown <= 0) {
            clearInterval(adTimer);
            skipTimer.style.display = 'none';
            skipButton.style.display = 'block';
        }
    }, 1000);
}

// Skip ad
function skipAd() {
    document.getElementById('adOverlay').style.display = 'none';
    document.getElementById('mainVideo').play();
    
    // Log skip event
    logAdEvent('skip');
}

// Click ad
function clickAd() {
    // Log click event
    logAdEvent('click');
    
    // In real app, this would open the advertiser's page
    alert('Ad clicked! In production, this would redirect to the advertiser.');
    
    // Close ad and resume video
    skipAd();
}

// Log ad events
function logAdEvent(eventType) {
    if (!currentAd) return;
    
    fetch('/api/report_click', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            ad_id: currentAd.id,
            event_type: eventType,
            predicted_ctr: currentAd.predicted_ctr,
            timestamp: new Date().toISOString()
        })
    });
}

// Close video player
function closeVideo() {
    document.getElementById('videoPlayerModal').style.display = 'none';
    const video = document.getElementById('mainVideo');
    video.pause();
    video.currentTime = 0;
    
    if (adTimer) {
        clearInterval(adTimer);
    }
}

// Chat functionality
let chatOpen = false;

function toggleChat() {
    chatOpen = !chatOpen;
    const chatBody = document.getElementById('chatBody');
    const chatToggle = document.getElementById('chatToggle');
    
    if (chatOpen) {
        chatBody.style.display = 'block';
        chatToggle.className = 'fas fa-chevron-down';
    } else {
        chatBody.style.display = 'none';
        chatToggle.className = 'fas fa-chevron-up';
    }
}

function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message
    addMessage(message, 'user');
    input.value = '';
    
    // Get bot response
    fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        addMessage(data.response || 'I can help you understand our ad campaigns. Try asking about CTR, performance metrics, or optimization strategies!', 'bot');
    })
    .catch(() => {
        addMessage('I can help you understand our ad campaigns. Try asking about CTR, performance metrics, or optimization strategies!', 'bot');
    });
}

function handleChatKeypress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

function addMessage(text, sender) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    messageDiv.innerHTML = `<p>${text}</p>`;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Initialize on load
document.addEventListener('DOMContentLoaded', function() {
    // Auto-open chat after 5 seconds for demo
    setTimeout(() => {
        if (!chatOpen) {
            toggleChat();
        }
    }, 5000);
});

// Handle video errors
document.getElementById('mainVideo').addEventListener('error', function() {
    console.log('Video error - using fallback');
    this.src = 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4';
});
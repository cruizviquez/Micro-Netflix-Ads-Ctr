// static/script.js
const videoPlayerContainer = document.getElementById('videoPlayerContainer');
const playingMovieTitle = document.getElementById('playingMovieTitle');
const movieVideo = document.getElementById('movieVideo');
const adOverlay = document.getElementById('adOverlay');
const adImage = document.getElementById('adImage');getUserProfile()
const adTitle = document.getElementById('adTitle');
const predictedCtr = document.getElementById('predictedCtr');
const clickAdBtn = document.getElementById('clickAdBtn');
const skipAdBtn = document.getElementById('skipAdBtn');

let currentPlayingMovie = null;
let currentAd = null;

function getUserProfile() {
    return {
        age: parseInt(document.getElementById('userAge').value),
        gender: document.getElementById('userGender').value,
        subscription_tier: document.getElementById('userSubscription').value,
        genre_preference: document.getElementById('userGenrePref').value,
        watch_hours: parseInt(document.getElementById('userAge').value) * 0.8 // Simple mock for watch hours
    };
}

async function playMovie(movieId, title, genre) {
    currentPlayingMovie = { id: movieId, title: title, genre: genre };
    playingMovieTitle.textContent = title;
    videoPlayerContainer.classList.remove('hidden');
    movieVideo.load(); // Reload video source if it changes
    movieVideo.play();

    // Simulate ad break after a few seconds
    setTimeout(showAd, 5000); // Show ad after 5 seconds of movie playback
}

async function showAd() {
    movieVideo.pause();
    adOverlay.classList.remove('hidden');

    const userProfile = getUserProfile();
    const contentContext = {
        genre: currentPlayingMovie.genre,
        time_of_day: new Date().getHours() // Example context
    };

    try {
        const response = await fetch('/simulate_ad_request', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ user_profile: userProfile, content_context: contentContext })
        });
        const adData = await response.json();

        if (response.ok) {
            currentAd = adData;
            adImage.src = adData.image;
            adTitle.textContent = adData.title;
            predictedCtr.textContent = adData.predicted_ctr;
            console.log("Predicted Ad:", adData);
        } else {
            console.error("Error fetching ad:", adData.error);
            // Fallback: show a generic ad or no ad
            adImage.src = "https://via.placeholder.com/640x360?text=Generic+Ad";
            adTitle.textContent = "Generic Advertisement";
            predictedCtr.textContent = "N/A";
            currentAd = { id: "generic", title: "Generic Ad", category: "Generic", predicted_ctr: "N/A" };
        }
    } catch (error) {
        console.error("Network error fetching ad:", error);
        adImage.src = "https://via.placeholder.com/640x360?text=Error+Loading+Ad";
        adTitle.textContent = "Ad Failed to Load";
        predictedCtr.textContent = "N/A";
        currentAd = { id: "error", title: "Ad Error", category: "Error", predicted_ctr: "N/A" };
    }
}

function closeAd() {
    adOverlay.classList.add('hidden');
    movieVideo.play();
    currentAd = null;
}

function closeVideoPlayer() {
    movieVideo.pause();
    movieVideo.currentTime = 0;
    videoPlayerContainer.classList.add('hidden');
    adOverlay.classList.add('hidden'); // Ensure ad is hidden too
    currentPlayingMovie = null;
    currentAd = null;
}

clickAdBtn.addEventListener('click', () => {
    if (currentAd) {
        logAdEvent('click', currentAd.id);
        alert(`Simulated click on Ad: "${currentAd.title}"!`);
    }
    closeAd();
});

skipAdBtn.addEventListener('click', () => {
    if (currentAd) {
        logAdEvent('skip', currentAd.id);
        alert(`Simulated skip on Ad: "${currentAd.title}"!`);
    }
    closeAd();
});

async function logAdEvent(eventType, adId) {
    const userProfile = getUserProfile();
    const contentContext = currentPlayingMovie ? { genre: currentPlayingMovie.genre } : {};
    try {
        await fetch('/ad_event', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                eventType: eventType,
                adId: adId,
                timestamp: new Date().toISOString(),
                userProfile: userProfile,
                contentContext: contentContext
            })
        });
        console.log(`Event ${eventType} for ad ${adId} logged.`);
    } catch (error) {
        console.error("Error logging ad event:", error);
    }
}


// --- LLM Chat Logic ---
const chatHistory = document.getElementById('chatHistory');
const llmInput = document.getElementById('llmInput');

function addChatMessage(message, sender) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', sender);
    messageElement.textContent = message;
    chatHistory.appendChild(messageElement);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Auto-scroll to bottom
}

async function askLLM() {
    const question = llmInput.value.trim();
    if (!question) return;

    addChatMessage(question, 'user');
    llmInput.value = '';
    llmInput.disabled = true; // Disable input while waiting for response

    try {
        addChatMessage("Thinking...", 'bot'); // Show a temporary "typing" message
        const response = await fetch('/ask_llm', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: question })
        });
        const data = await response.json();
        chatHistory.lastChild.remove(); // Remove "Thinking..."
        addChatMessage(data.answer, 'bot');
    } catch (error) {
        console.error("Error asking LLM:", error);
        chatHistory.lastChild.remove();
        addChatMessage("Sorry, I couldn't get a response. Please try again.", 'bot');
    } finally {
        llmInput.disabled = false;
        llmInput.focus();
    }
}

function handleLLMInput(event) {
    if (event.key === 'Enter') {
        askLLM();
    }
}

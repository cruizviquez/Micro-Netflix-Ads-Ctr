// Global variables
let selectedMovie = null;
let selectedMovieGenre = null;

// Select a movie
function selectMovie(movieId, genre) {
    // Remove previous selection
    document.querySelectorAll('.movie-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    // Add selection to clicked movie
    const clickedCard = document.querySelector(`[data-movie-id="${movieId}"]`);
    if (clickedCard) {
        clickedCard.classList.add('selected');
    }
    
    selectedMovie = movieId;
    selectedMovieGenre = genre;
}

// Request optimized ad
function requestOptimizedAd() {
    if (!selectedMovie) {
        alert('Please select a movie first!');
        return;
    }
    
    // Get form data
    const formData = new FormData(document.getElementById('userProfileForm'));
    const userProfile = {};
    
    for (let [key, value] of formData.entries()) {
        userProfile[key] = isNaN(value) ? value : parseInt(value);
    }
    
    // Prepare request data
    const requestData = {
        user_profile: userProfile,
        content_context: {
            movie_id: selectedMovie,
            genre: selectedMovieGenre
        }
    };
    
    // Show loading
    showResults();
    document.getElementById('adResult').innerHTML = '<div class="loading">Finding the best ad for you...</div>';
    
    // Make API request
    fetch('/api/optimize_ad', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayAdResult(data.ad);
            displayAlternatives(data.alternatives);
        } else {
            document.getElementById('adResult').innerHTML = `<div class="error">${data.error}</div>`;
        }
    })
    .catch(error => {
        document.getElementById('adResult').innerHTML = `<div class="error">Error: ${error.message}</div>`;
    });
}

// Display ad result
function displayAdResult(ad) {
    const adResultHTML = `
        <div class="ad-display">
            <img src="${ad.image}" alt="${ad.title}">
            <h3>${ad.title}</h3>
            <div class="personalized-copy">${ad.personalized_copy || ''}</div>
        </div>
        <div class="ad-info">
            <div class="info-item">
                <div class="info-label">Predicted CTR</div>
                <div class="ctr-value">${(ad.predicted_ctr * 100).toFixed(2)}%</div>
            </div>
            <div class="info-item">
                <div class="info-label">Category</div>
                <div>${ad.category}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Freshness Score</div>
                <div>${(ad.creative_freshness * 100).toFixed(0)}%</div>
            </div>
            <div class="info-item">
                <div class="info-label">Overall Score</div>
                <div>${(ad.score * 100).toFixed(2)}%</div>
            </div>
        </div>
        ${ad.insights ? `<div class="insights"><strong>Insights:</strong> ${ad.insights}</div>` : ''}
        <div class="ad-actions">
            <button onclick="simulateClick('${ad.id}')" class="btn-primary">Simulate Click</button>
        </div>
    `;
    
    document.getElementById('adResult').innerHTML = adResultHTML;
}

// Display alternative ads
function displayAlternatives(alternatives) {
    if (!alternatives || alternatives.length === 0) return;
    
    let altHTML = '<h3>Alternative Ads</h3><div class="alternatives-grid">';
    
    alternatives.forEach(alt => {
        altHTML += `
            <div class="alt-ad">
                <strong>${alt.ad.title}</strong>
                <span>CTR: ${(alt.predicted_ctr * 100).toFixed(2)}%</span>
            </div>
        `;
    });
    
    altHTML += '</div>';
    document.getElementById('alternatives').innerHTML = altHTML;
}

// Show results section
function showResults() {
    document.getElementById('results').style.display = 'block';
    
    // Display selected movie info
    const movieInfo = `
        <strong>Selected Content:</strong> Movie ID ${selectedMovie} (${selectedMovieGenre})
    `;
    document.getElementById('selectedMovie').innerHTML = movieInfo;
}

// Simulate ad click
function simulateClick(adId) {
    const clickData = {
        ad_id: adId,
        user_id: 'demo_user_' + Date.now(),
        session_id: 'session_' + Date.now()
    };
    
    fetch('/api/report_click', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(clickData)
    })
    .then(response => response.json())
    .then(data => {
        alert('Click recorded! Ad performance data updated.');
    })
    .catch(error => {
        console.error('Error recording click:', error);
    });
}

// Test LLM functionality
function testLLM() {
    fetch('/api/test_llm')
        .then(response => response.json())
        .then(data => {
            alert(`LLM Test Result:\n\nGenerated Ad: ${data.generated_ad}\n\nStatus: ${data.status}`);
        })
        .catch(error => {
            alert(`Error testing LLM: ${error.message}`);
        });
}

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    // Auto-select first movie for demo purposes
    const firstMovie = document.querySelector('.movie-card');
    if (firstMovie) {
        const movieId = firstMovie.getAttribute('data-movie-id');
        const genre = firstMovie.querySelector('.genre-tag').textContent;
        selectMovie(movieId, genre);
    }
    
    // Add event listeners for form changes
    const formInputs = document.querySelectorAll('#userProfileForm input, #userProfileForm select');
    formInputs.forEach(input => {
        input.addEventListener('change', function() {
            // Could trigger real-time predictions here
            console.log('User profile updated:', this.name, this.value);
        });
    });
});

// Additional utility functions for dashboard
function refreshData() {
    location.reload();
}

function exportData() {
    // Simulate data export
    const data = {
        timestamp: new Date().toISOString(),
        metrics: 'Dashboard metrics would be exported here'
    };
    
    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = 'ctr-report-' + Date.now() + '.json';
    link.click();
    
    URL.revokeObjectURL(url);
}

// Test functions for test page
function testWithSampleUser() {
    console.log('Testing with sample user...');
}

function generateTestData() {
    console.log('Generating test data...');
}
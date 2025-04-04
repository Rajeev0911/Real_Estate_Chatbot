// document.getElementById("mic-btn").addEventListener("click", () => {
//     const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
//     recognition.lang = "en-US";
//     recognition.start();
//     recognition.onresult = (event) => {
//         const transcript = event.results[0][0].transcript;
//         const chatContainer = document.getElementById("chat-container");
//         const userMessage = document.createElement("div");
//         userMessage.classList.add("message", "user");
//         userMessage.textContent = transcript;
//         chatContainer.appendChild(userMessage);
//         chatContainer.scrollTop = chatContainer.scrollHeight;
//     };
//     recognition.onerror = (event) => {
//         alert(`Error: ${event.error}`);
//     };
// });



// Voice recognition functionality
document.getElementById("mic-btn").addEventListener("click", () => {
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = "en-US";
    recognition.start();
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        addUserMessage(transcript);
    };
    recognition.onerror = (event) => {
        alert(`Error: ${event.error}`);
    };
});

// Text input functionality
document.getElementById("send-btn").addEventListener("click", sendMessage);
document.getElementById("user-input").addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
        sendMessage();
    }
});

function sendMessage() {
    const userInput = document.getElementById("user-input");
    const message = userInput.value.trim();
    
    if (message) {
        addUserMessage(message);
        userInput.value = "";
    }
}

function addUserMessage(text) {
    const chatContainer = document.getElementById("chat-container");
    const userMessage = document.createElement("div");
    userMessage.classList.add("message", "user");
    userMessage.textContent = text;
    chatContainer.appendChild(userMessage);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    // Add a loading message
    const loadingMessage = document.createElement("div");
    loadingMessage.classList.add("message", "agent");
    loadingMessage.textContent = "Processing your request...";
    loadingMessage.id = "loading-message";
    chatContainer.appendChild(loadingMessage);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    // Make API call to the backend
    fetch('/api/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: text })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Remove the loading message
        document.getElementById("loading-message").remove();
        
        // Add the actual response
        const agentMessage = document.createElement("div");
        agentMessage.classList.add("message", "agent");
        
        // Format the response for better readability
        const formattedResponse = formatResponseText(data.response);
        agentMessage.innerHTML = formattedResponse;
        
        chatContainer.appendChild(agentMessage);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
        // Speak the response if audio is enabled
        speakResponse(data.response);
    })
    .catch(error => {
        // Remove the loading message
        if (document.getElementById("loading-message")) {
            document.getElementById("loading-message").remove();
        }
        
        // Show error message
        const errorMessage = document.createElement("div");
        errorMessage.classList.add("message", "agent");
        errorMessage.textContent = "Sorry, there was an error processing your request. Please try again.";
        chatContainer.appendChild(errorMessage);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        console.error('Error:', error);
    });
}

// Format the response text with proper line breaks and highlights
function formatResponseText(text) {
    // Replace property titles with highlighted versions
    let formattedText = text.replace(/Property \d+:/g, match => `<strong>${match}</strong>`);
    
    // Replace line breaks with HTML breaks
    formattedText = formattedText.replace(/\n/g, '<br>');
    
    // Highlight important information
    formattedText = formattedText.replace(/- ([^:]+):/g, '- <strong>$1</strong>:');
    
    return formattedText;
}

// Audio output functionality
let isSpeaking = false;
const synth = window.speechSynthesis;

document.getElementById("speaker-btn").addEventListener("click", () => {
    isSpeaking = !isSpeaking;
    const speakerBtn = document.getElementById("speaker-btn");
    speakerBtn.textContent = isSpeaking ? "🔇" : "🔊";
    
    if (!isSpeaking) {
        synth.cancel();
    }
});

function speakResponse(text) {
    if (!isSpeaking) return;
    
    // Clean up the text for speaking (remove property details)
    const cleanText = text.split("Property 1:")[0];
    
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    synth.speak(utterance);
}
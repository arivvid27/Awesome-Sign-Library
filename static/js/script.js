document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const detectedLetterElement = document.getElementById('detected-letter');
    const letterBufferElement = document.getElementById('letter-buffer');
    const processedTextElement = document.getElementById('processed-text');
    const addLetterBtn = document.getElementById('add-letter-btn');
    const spaceBtn = document.getElementById('space-btn');
    const clearBtn = document.getElementById('clear-btn');
    const processBtn = document.getElementById('process-btn');
    
    // Global variables
    let currentLetter = '';
    let letterBuffer = '';
    let processingActive = false;
    
    // Initialize
    updateBuffer();
    
    // Poll for current letter detection
    setInterval(pollCurrentLetter, 500);
    
    // Event listeners
    addLetterBtn.addEventListener('click', addCurrentLetter);
    spaceBtn.addEventListener('click', addSpace);
    clearBtn.addEventListener('click', clearBuffer);
    processBtn.addEventListener('click', processText);
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(event) {
        if (event.code === 'Space') {
            addCurrentLetter();
        } else if (event.code === 'Enter') {
            processText();
        } else if (event.code === 'Escape') {
            clearBuffer();
        }
    });
    
    // Functions
    function pollCurrentLetter() {
        fetch('/get_letter')
            .then(response => response.json())
            .then(data => {
                currentLetter = data.letter;
                updateDetectedLetter();
            })
            .catch(error => console.error('Error fetching current letter:', error));
    }
    
    function updateDetectedLetter() {
        detectedLetterElement.textContent = currentLetter || '-';
    }
    
    function addCurrentLetter() {
        if (!currentLetter || processingActive) return;
        
        fetch('/add_letter')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    letterBuffer = data.buffer;
                    updateBuffer();
                    
                    // Visual feedback
                    addLetterBtn.classList.add('active');
                    setTimeout(() => {
                        addLetterBtn.classList.remove('active');
                    }, 200);
                }
            })
            .catch(error => console.error('Error adding letter:', error));
    }
    
    function addSpace() {
        if (processingActive) return;
        
        letterBuffer += ' ';
        updateBuffer();
        
        // Visual feedback
        spaceBtn.classList.add('active');
        setTimeout(() => {
            spaceBtn.classList.remove('active');
        }, 200);
    }
    
    function clearBuffer() {
        if (processingActive) return;
        
        fetch('/clear_buffer')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    letterBuffer = '';
                    updateBuffer();
                    
                    // Visual feedback
                    clearBtn.classList.add('active');
                    setTimeout(() => {
                        clearBtn.classList.remove('active');
                    }, 200);
                }
            })
            .catch(error => console.error('Error clearing buffer:', error));
    }
    
    function processText() {
        if (!letterBuffer || processingActive) return;
        
        processingActive = true;
        processBtn.textContent = 'Processing...';
        processBtn.disabled = true;
        
        fetch('/process_text')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    processedTextElement.textContent = data.processed_text;
                    letterBuffer = '';
                    updateBuffer();
                } else {
                    console.error('Error processing text:', data.message);
                }
            })
            .catch(error => console.error('Error processing text:', error))
            .finally(() => {
                processingActive = false;
                processBtn.textContent = 'Process to Words';
                processBtn.disabled = false;
            });
    }
    
    function updateBuffer() {
        letterBufferElement.textContent = letterBuffer || 'No letters added yet';
    }
    
    // Periodically sync buffer with server
    setInterval(() => {
        fetch('/get_buffer')
            .then(response => response.json())
            .then(data => {
                letterBuffer = data.buffer;
                updateBuffer();
            })
            .catch(error => console.error('
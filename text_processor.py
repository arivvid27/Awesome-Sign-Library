import os
import google.generativeai as genai

class TextProcessor:
    def __init__(self):
        """Initialize the text processor with Gemini AI"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Get available models
        self.model = genai.GenerativeModel('gemini-pro')
        
    def process_text(self, letter_stream):
        """Process a stream of ASL letters into words using Gemini AI"""
        if not letter_stream:
            return ""
        
        # Construct the prompt for Gemini AI
        prompt = f"""
        I have a stream of ASL letters without spaces: "{letter_stream}"
        Please break this into a coherent sentence or phrase with proper spacing and punctuation.
        If certain combinations don't make sense, make reasonable corrections or guesses.
        Only respond with the final processed text, no explanations.
        """
        
        # Generate response from Gemini
        response = self.model.generate_content(prompt)
        
        # Extract and return the processed text
        if response and response.text:
            return response.text.strip()
        
        # Fallback if Gemini fails
        return self._simple_fallback_processing(letter_stream)
    
    def _simple_fallback_processing(self, letter_stream):
        """Fallback processing method if Gemini fails"""
        # This is a very basic fallback that just returns the letter stream 
        # with some common words detected
        common_words = {
            "THE", "AND", "YOU", "THAT", "WAS", "FOR", "ARE", "WITH", "HIS", "THEY",
            "THIS", "HAVE", "FROM", "ONE", "HAD", "WORD", "BUT", "NOT", "WHAT", "ALL",
            "WERE", "WHEN", "YOUR", "SAID", "THERE", "USE", "EACH", "WHICH", "SHE", "HOW",
            "THEIR", "WILL", "OTHER", "ABOUT", "OUT", "MANY", "THEN", "THEM", "THESE", "SOME",
            "HER", "WOULD", "MAKE", "LIKE", "HIM", "INTO", "TIME", "HAS", "LOOK", "TWO",
            "MORE", "WRITE", "SEE", "NUMBER", "WAY", "COULD", "PEOPLE", "THAN", "FIRST", "WATER"
        }
        
        result = ""
        buffer = ""
        
        for letter in letter_stream:
            buffer += letter
            
            # Check if buffer matches any common words
            for word in common_words:
                if word in buffer:
                    result += buffer.replace(word, word + " ")
                    buffer = ""
                    break
        
        # Add any remaining letters
        result += buffer
        
        return result
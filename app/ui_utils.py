import streamlit as st
import streamlit.components.v1 as components
import time

def render_live_timer(label_text, start_time=None):
    """
    Renders a real-time counting timer using a Streamlit component (iframe).
    This ensures JavaScript execution which might be blocked in st.markdown.
    Returns a Streamlit placeholder that can be updated/cleared later.
    """
    if start_time is None:
        start_time = time.time()
    
    placeholder = st.empty()
    
    # Calculate elapsed time so far to initialize the timer correctly
    elapsed_server = time.time() - start_time
    
    # HTML/JS code to run inside the iframe
    # We use basic CSS to mimic Streamlit's look
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600&family=Source+Code+Pro:wght@400;600&display=swap');
            
            body {{
                font-family: 'Source Sans Pro', sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                align-items: center;
                background-color: transparent;
                color: rgb(49, 51, 63); /* Default Streamlit Light Mode Text */
            }}
            
            .timer-box {{
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 16px; /* Match Streamlit default */
            }}
            
            .timer-val {{
                font-family: 'Source Code Pro', monospace; 
                font-weight: bold; 
                color: #FF4B4B; 
                background-color: #f0f2f6; 
                padding: 2px 6px; 
                border-radius: 4px;
            }}
            
            /* Dark Mode Adjustments */
            @media (prefers-color-scheme: dark) {{
                body {{
                    color: #fafafa;
                }}
                .timer-val {{
                    background-color: #262730;
                    color: #FF4B4B;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="timer-box">
            <span>{label_text}</span>
            <span id="timer" class="timer-val">{elapsed_server:.1f}s</span>
        </div>
        <script>
            // Initialize start time based on server elapsed time
            var start = Date.now() - ({elapsed_server} * 1000);
            var el = document.getElementById("timer");
            
            setInterval(function() {{
                var now = Date.now();
                var elapsed = ((now - start) / 1000).toFixed(1);
                el.innerText = elapsed + "s";
            }}, 100);
        </script>
    </body>
    </html>
    """
    
    # Render the component inside the placeholder
    # height=50 ensures enough space for the text without scrollbars
    with placeholder:
        components.html(html_code, height=40)
        
    return placeholder

def display_token_usage(response):
    """
    Displays token usage information from the LLM response.
    """
    if hasattr(response, "response_metadata"):
        token_usage = response.response_metadata.get("token_usage")
        if token_usage:
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0)
            
            st.caption(f"🪙 Token Usage: Input: {prompt_tokens} | Output: {completion_tokens} | Total: {total_tokens}")

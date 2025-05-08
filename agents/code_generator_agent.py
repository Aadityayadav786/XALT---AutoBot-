def generate_embed_code(live_url):
    return f"""
<!-- Xalt Chatbot Integration -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
  #chatbot-button {{ position: fixed; bottom: 20px; right: 20px; background-color: #4CAF50; color: white; border: none; border-radius: 50%; padding: 15px; cursor: pointer; z-index: 1000; }}
  #chatbot-frame {{ display: none; position: fixed; bottom: 80px; right: 20px; width: 400px; height: 600px; border: none; z-index: 1000; box-shadow: 0px 0px 10px rgba(0,0,0,0.3); }}
</style>
<button id="chatbot-button"><i class="fas fa-comment"></i></button>
<iframe id="chatbot-frame" src="{live_url}" title="Xalt Chatbot"></iframe>
<script>
  const btn = document.getElementById('chatbot-button');
  const frame = document.getElementById('chatbot-frame');
  btn.addEventListener('click', () => {{
    frame.style.display = frame.style.display === 'none' ? 'block' : 'none';
  }});
</script>
"""



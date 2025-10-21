# 🤖 OpenAI FAQ Chatbot - Complete Setup Guide

## 🎯 What This Bot Does

✅ **Answers FAQ questions** - Instantly from your database (FREE)  
✅ **Answers ANY question** - Uses OpenAI GPT when needed  
✅ **Maintains context** - Remembers conversation history  
✅ **Saves money** - Uses FAQs first, OpenAI only when necessary  
✅ **Professional responses** - GPT-3.5-turbo quality answers  

---

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **OpenAI Account** with API key
3. **Payment method** added to OpenAI (minimum $5)

---

## 🚀 Step-by-Step Setup

### **Step 1: Get OpenAI API Key**

1. Go to [OpenAI Platform](https://platform.openai.com/signup)
2. Sign up or log in
3. Add payment method:
   - Click your profile → "Billing"
   - Add payment (credit card)
   - Add minimum $5 credit
4. Generate API key:
   - Go to [API Keys](https://platform.openai.com/api-keys)
   - Click "Create new secret key"
   - **Name it:** "FAQ Chatbot"
   - Copy the key (starts with `sk-...`)
   - ⚠️ **Save it immediately!** You can't see it again

### **Step 2: Install Dependencies**

```bash
# Install all required packages
pip install streamlit nltk scikit-learn numpy openai

# Or use requirements file:
pip install -r requirements.txt
```

### **Step 3: Create Project Files**

Create a folder structure:
```
openai_faq_bot/
├── app.py              # Main app (save artifact code here)
├── faqs.txt           # FAQ data (auto-created)
└── requirements.txt   # Dependencies
```

### **Step 4: Run the App**

```bash
# Navigate to your folder
cd openai_faq_bot

# Run Streamlit
streamlit run app.py
```

### **Step 5: Configure in Browser**

1. App opens in browser automatically
2. In sidebar, paste your API key
3. Click outside the input box
4. You'll see "✅ API key configured"
5. Start chatting!

---

## 💰 Pricing & Costs

### **OpenAI Pricing (GPT-3.5-turbo):**
- **Input:** $0.0015 per 1K tokens (~750 words)
- **Output:** $0.002 per 1K tokens (~750 words)

### **Real Cost Examples:**

| Usage | Estimated Cost |
|-------|----------------|
| 100 questions | $0.20 - $0.50 |
| 500 questions | $1.00 - $2.50 |
| 1,000 questions | $2.00 - $5.00 |

### **How to Save Money:**

1. ✅ **Use FAQ matching** (default enabled)
   - FAQ answers = FREE
   - Only pays for non-FAQ questions
   
2. ✅ **Add common questions to FAQs**
   - If users ask same thing, add to `faqs.txt`
   - Future answers are FREE
   
3. ✅ **Set appropriate threshold**
   - Default 0.3 is good balance
   - Higher = Uses more FAQs (cheaper)
   
4. ✅ **Monitor usage**
   - Check "Est. Cost" in sidebar
   - View usage at [OpenAI Usage](https://platform.openai.com/usage)

---

## 🎮 How to Use the App

### **Asking Questions:**

**FAQ Questions (FREE):**
```
You: What are your business hours?
Bot: Our business hours are Monday to Friday, 9 AM to 6 PM...
     ✅ FAQ Database (Free)
     🎯 Confidence: 95%
```

**General Questions (Uses OpenAI):**
```
You: What is machine learning?
Bot: Machine learning is a subset of AI that enables...
     🤖 OpenAI GPT
     🎯 Confidence: 85%
```

**Follow-up Questions:**
```
You: Can you explain more about that?
Bot: [Maintains context from previous question]
     🤖 OpenAI GPT
```

### **Managing FAQs:**

**Add New FAQ:**
1. Sidebar → "Add New FAQ"
2. Enter question and answer
3. Click "Add FAQ"
4. ✅ Saved to `faqs.txt`

**View All FAQs:**
1. Sidebar → "View All FAQs"
2. Browse your FAQ database
3. See all questions and answers

**Edit FAQs Manually:**
1. Open `faqs.txt` in any text editor
2. Format: `Question | Answer`
3. Save file
4. Click "🔄 Reload FAQs" in app

---

## ⚙️ Settings Explained

### **Enable FAQ Matching:**
- ✅ **ON (Recommended):** Checks FAQs first, saves money
- ❌ **OFF:** Uses OpenAI for everything (more expensive)

### **FAQ Match Threshold:**
- **0.1-0.2:** Very lenient, uses OpenAI more
- **0.3-0.4:** Balanced (recommended)
- **0.5-0.7:** Very strict, uses FAQs more

### **Response Creativity:**
- **0.0-0.3:** More focused, factual
- **0.4-0.7:** Balanced (default)
- **0.8-1.0:** More creative, varied

---

## 📊 Understanding the Interface

### **Sidebar Stats:**

**Total FAQs:** Number of questions in your database

**Messages:** Total chat exchanges

**Est. Tokens Used:** Approximate API usage

**Est. Cost:** Approximate spending (tokens × $0.002)

### **Message Indicators:**

🟢 **"✅ FAQ Database (Free)"** - Answered from FAQs (no cost)

🔵 **"🤖 OpenAI GPT"** - Answered by AI (costs tokens)

📌 **"Matched: [question]"** - Shows which FAQ matched

🎯 **"Confidence: 85%"** - Match certainty

---

## 🔧 Customization

### **Change Company Info:**

Edit `faqs.txt` to include your specific information:
```
What is your company name? | We are XYZ Corporation, founded in 2020.

What products do you sell? | We specialize in AI-powered solutions...

Where are you located? | Our headquarters is at 123 Main St, City, State.
```

### **Adjust System Prompt:**

In the code (line ~156), modify the system message:
```python
"content": f"""You are [YOUR COMPANY] customer service assistant.

[Add your company guidelines here]

Our company values:
- Customer satisfaction
- Quick responses
- Professional tone
"""
```

### **Change AI Model:**

Line 192, change model:
```python
model="gpt-3.5-turbo",  # Fast & cheap
# OR
model="gpt-4",          # Better quality, 10x more expensive
```

---

## 🐛 Troubleshooting

### **"Invalid API key" Error:**

**Solution:**
- Check key starts with `sk-`
- No spaces before/after
- Key is active on [OpenAI Platform](https://platform.openai.com/api-keys)
- Account has credits

### **"Rate limit exceeded" Error:**

**Solution:**
- You've hit free tier limits
- Wait 1 minute and try again
- Upgrade to paid tier
- Check [Usage Page](https://platform.openai.com/usage)

### **"Insufficient credits" Error:**

**Solution:**
- Go to [Billing](https://platform.openai.com/account/billing)
- Add more credits ($5 minimum)
- Wait 5-10 minutes for activation

### **Slow Responses:**

**Possible causes:**
- Internet connection slow
- OpenAI servers busy
- Try again in a moment

### **FAQ Not Matching:**

**Solutions:**
- Lower the threshold (sidebar slider)
- Add more similar phrasings to FAQs
- Check `faqs.txt` format is correct

---

## 📁 File Structure

```
openai_faq_bot/
│
├── app.py                 # Main Streamlit application
│   ├── File operations   # Load/save FAQs
│   ├── OpenAI class      # ChatGPT integration
│   └── Streamlit UI      # Web interface
│
├── faqs.txt              # Your FAQ database
│   └── Format: Question | Answer
│
└── requirements.txt      # Python dependencies
```

---

## 🔐 Security Best Practices

### **Protect Your API Key:**

1. **Never commit to GitHub:**
```bash
# Add to .gitignore:
echo "*.env" >> .gitignore
echo "api_key.txt" >> .gitignore
```

2. **Use environment variables:**
```bash
# In terminal:
export OPENAI_API_KEY="sk-..."

# In code:
import os
api_key = os.getenv("OPENAI_API_KEY")
```

3. **Set spending limits:**
- [OpenAI Billing](https://platform.openai.com/account/billing) → Limits
- Set monthly budget (e.g., $10)
- Get email alerts

### **Monitor Usage:**

Check regularly:
- [Usage Dashboard](https://platform.openai.com/usage)
- View daily/monthly spending
- See which days had high usage

---

## 🚀 Advanced Features

### **Add Rate Limiting:**

```python
import time

# Track requests
if 'last_request_time' not in st.session_state:
    st.session_state.last_request_time = 0

# Limit to 1 request per 2 seconds
current_time = time.time()
if current_time - st.session_state.last_request_time < 2:
    st.warning("Please wait before asking another question")
else:
    st.session_state.last_request_time = current_time
    # Process question
```

### **Add User Authentication:**

```python
# Simple password protection
password = st.text_input("Enter password:", type="password")

if password != "your_password":
    st.error("Invalid password")
    st.stop()
```

### **Export Chat History:**

```python
if st.button("📥 Download Chat History"):
    chat_text = "\n\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in st.session_state.messages
    ])
    st.download_button(
        "Download",
        chat_text,
        "chat_history.txt"
    )
```

---

## 📈 Optimization Tips

### **1. Build a Strong FAQ Database:**
- Add 20-30 common questions
- Cover all your products/services
- Include variations of questions
- Update regularly based on user questions

### **2. Monitor and Improve:**
- Check which questions use OpenAI
- Add popular questions to FAQs
- Review unusual questions
- Refine answers based on feedback

### **3. Context Management:**
- Keep conversations focused
- Clear chat for new topics
- Limit history to 6 messages (already implemented)
- Reduces token usage

### **4. Optimize Responses:**
- Set max_tokens to control length
- Use temperature 0.5-0.7 for balance
- Test different models for cost/quality

---

## 📱 Deployment Options

### **Option 1: Local Network (Team Access)**

```bash
# Run on local network
streamlit run app.py --server.address 0.0.0.0 --server.port 8501

# Access from other devices:
# http://YOUR_IP_ADDRESS:8501
```

**Find your IP:**
```bash
# Windows:
ipconfig

# Mac/Linux:
ifconfig
```

### **Option 2: Streamlit Cloud (Free Hosting)**

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secrets (API key):
   - Click "Advanced settings"
   - Add: `OPENAI_API_KEY = "sk-..."`
5. Deploy!

**Update code for secrets:**
```python
# In app.py, replace API key input with:
import streamlit as st

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    api_key = st.text_input("API Key:", type="password")
```

### **Option 3: Docker Container**

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

Run:
```bash
docker build -t faq-chatbot .
docker run -p 8501:8501 faq-chatbot
```

---

## 🎓 Example Use Cases

### **Customer Support:**
```
User: How do I return a product?
Bot: [From FAQ] We accept returns within 30 days...

User: What if it's been 35 days?
Bot: [OpenAI] While our standard policy is 30 days, 
     I recommend contacting our support team...
```

### **Product Information:**
```
User: What's the difference between Plan A and Plan B?
Bot: [From FAQ] Plan A includes... Plan B includes...

User: Which one is better for small businesses?
Bot: [OpenAI] For small businesses, Plan B might be better 
     because it offers...
```

### **General Knowledge:**
```
User: What is API?
Bot: [OpenAI] An API (Application Programming Interface) 
     is a set of protocols...

User: Do you have an API?
Bot: [From FAQ] Yes, we offer a REST API for developers...
```

---

## 🧪 Testing Your Bot

### **Test Checklist:**

**✅ FAQ Questions:**
- [ ] Test exact FAQ questions
- [ ] Test variations of FAQ questions
- [ ] Verify answers come from FAQ database
- [ ] Check confidence scores

**✅ Non-FAQ Questions:**
- [ ] Test general knowledge questions
- [ ] Test follow-up questions
- [ ] Verify context is maintained
- [ ] Check OpenAI responses are relevant

**✅ Edge Cases:**
- [ ] Test empty messages
- [ ] Test very long questions
- [ ] Test special characters
- [ ] Test multiple languages (if needed)

**✅ Functionality:**
- [ ] Add new FAQ works
- [ ] View FAQs displays correctly
- [ ] Reload FAQs updates properly
- [ ] Clear chat resets conversation
- [ ] Token counter is accurate

### **Sample Test Questions:**

```python
# FAQ Tests (should use FAQ database):
"What are your business hours?"
"when do you open"
"what time are you open"
"business hours?"

# General Tests (should use OpenAI):
"What is Python programming?"
"Tell me about climate change"
"How does photosynthesis work?"

# Context Tests:
"What are your hours?" → "Are you open on Sunday?"
"How do I track orders?" → "What if I don't have tracking number?"

# Edge Cases:
"" (empty)
"askjdhaksjdhaksjdh" (gibberish)
"你好" (other language)
```

---

## 📊 Monitoring & Analytics

### **Track Important Metrics:**

**Add to sidebar:**
```python
# Track FAQ vs OpenAI usage
if 'faq_count' not in st.session_state:
    st.session_state.faq_count = 0
if 'openai_count' not in st.session_state:
    st.session_state.openai_count = 0

# Display ratio
st.metric("FAQ Answers", st.session_state.faq_count)
st.metric("OpenAI Answers", st.session_state.openai_count)

if st.session_state.faq_count + st.session_state.openai_count > 0:
    faq_percentage = (st.session_state.faq_count / 
                     (st.session_state.faq_count + st.session_state.openai_count)) * 100
    st.metric("Cost Savings", f"{faq_percentage:.1f}%")
```

### **Log Unanswered Questions:**

```python
# Create a log file for questions that needed OpenAI
def log_openai_question(question, answer):
    with open('openai_questions.log', 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now()}: {question}\n")
```

Review this log weekly to:
- Find common questions to add to FAQs
- Identify gaps in your FAQ coverage
- Improve answer quality

---

## 🔄 Maintenance & Updates

### **Weekly Tasks:**

1. **Review Logs:**
   - Check `openai_questions.log`
   - Identify frequently asked questions
   - Add them to `faqs.txt`

2. **Update FAQs:**
   - Add new products/services
   - Update outdated information
   - Improve unclear answers

3. **Check Costs:**
   - Review [OpenAI Usage](https://platform.openai.com/usage)
   - Compare to previous weeks
   - Adjust FAQ threshold if needed

### **Monthly Tasks:**

1. **Analyze Performance:**
   - FAQ coverage percentage
   - Average response time
   - User satisfaction (if tracked)

2. **Optimize:**
   - Remove unused FAQs
   - Consolidate similar questions
   - Update system prompt if needed

3. **Backup:**
   - Save `faqs.txt` backup
   - Export chat logs if important
   - Document any custom changes

---

## 🆘 Support & Resources

### **Official Documentation:**
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Streamlit Docs](https://docs.streamlit.io)
- [NLTK Documentation](https://www.nltk.org)

### **Community:**
- [OpenAI Community](https://community.openai.com)
- [Streamlit Forum](https://discuss.streamlit.io)
- Stack Overflow with tags: `openai`, `streamlit`

### **Get Help:**
- OpenAI Support: support@openai.com
- Check [OpenAI Status](https://status.openai.com)
- Review error messages carefully

---

## 🎉 Quick Start Checklist

Follow this checklist to get started:

- [ ] **Step 1:** Get OpenAI API key from platform.openai.com
- [ ] **Step 2:** Add $5 credit to your account
- [ ] **Step 3:** Install requirements: `pip install -r requirements.txt`
- [ ] **Step 4:** Save app.py from artifact
- [ ] **Step 5:** Run: `streamlit run app.py`
- [ ] **Step 6:** Enter API key in sidebar
- [ ] **Step 7:** Test with FAQ question: "What are your hours?"
- [ ] **Step 8:** Test with general question: "What is AI?"
- [ ] **Step 9:** Add your own FAQs to `faqs.txt`
- [ ] **Step 10:** Monitor usage and costs

---

## 💡 Pro Tips

1. **Start Small:** Begin with 10-15 FAQs, expand gradually

2. **Quality Over Quantity:** Better to have 20 great FAQs than 100 poor ones

3. **Use Clear Questions:** Write FAQs as users would ask them

4. **Test Variations:** Add common misspellings and phrasings

5. **Monitor First Week:** Watch usage closely, adjust threshold

6. **Set Budget Alerts:** Prevent surprise bills

7. **Regular Updates:** Keep FAQs current with your business

8. **Save Good AI Answers:** Convert frequent OpenAI answers to FAQs

9. **Clear Instructions:** Update system prompt for your specific needs

10. **User Feedback:** Add feedback buttons to improve responses

---

## 🚀 You're Ready!

Your chatbot is now ready to answer ANY question:
- ✅ FAQs answered instantly (FREE)
- ✅ Everything else answered by GPT (pennies per query)
- ✅ Professional, accurate responses
- ✅ Easy to customize and maintain

**Run this command to start:**
```bash
streamlit run app.py
```

**Questions? Issues? Check the troubleshooting section above!**

Good luck with your AI-powered FAQ chatbot! 🎉
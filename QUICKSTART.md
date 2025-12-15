# 🚀 Quick Start Guide

## 5-Minute Setup

### Step 1: Install Python
Make sure you have Python 3.8+ installed:
```bash
python --version
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal.

### Step 4: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This takes 5-10 minutes. ☕

### Step 5: Run the App
```bash
streamlit run app.py
```

Your browser will open automatically! 🎉

## 🎯 Try These Examples

1. **Market Overview** - Already loaded
2. **Stock Analysis** - Enter `AAPL`
3. **Technical Analysis** - Enter `MSFT`
4. **ML Predictions** - Enter `GOOGL` (wait 1-2 min)
5. **Sentiment** - Enter `TSLA`

## 🔧 Troubleshooting

**App won't start?**
```bash
streamlit run app.py --server.port 8502
```

**Module not found?**
```bash
pip install --upgrade -r requirements.txt
```

**FinBERT slow?**
- First load downloads model (~500MB)
- Subsequent loads are instant

## 💡 Tips

- Use `Ctrl+C` to stop the app
- Type `deactivate` to exit virtual environment
- Read full README.md for details

Happy analyzing! 📈

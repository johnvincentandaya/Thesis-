# Mobile Device Setup Guide

## Problem
The backend is not connected when using the app on mobile devices because mobile devices can't connect to `localhost`.

## Solution
The app now automatically detects the correct IP address for mobile devices. Here's how to set it up:

### Step 1: Find Your Computer's IP Address

#### Windows:
1. Open Command Prompt
2. Type: `ipconfig`
3. Look for "IPv4 Address" under your network adapter (usually starts with 192.168.x.x or 10.x.x.x)

#### Mac:
1. Open Terminal
2. Type: `ifconfig | grep "inet " | grep -v 127.0.0.1`
3. Look for your local IP address (usually starts with 192.168.x.x or 10.x.x.x)

#### Linux:
1. Open Terminal
2. Type: `ip addr show` or `ifconfig`
3. Look for your local IP address

### Step 2: Start the Backend Server
Make sure the backend is running on your computer:
```bash
cd backend
python app.py
```

The backend should show:
```
Server will be available at http://127.0.0.1:5001
```

### Step 3: Start the Frontend with Mobile Access
Instead of using `npm start`, use:
```bash
# Option 1: Start with your computer's IP
npm start -- --host 0.0.0.0

# Option 2: Or set the host explicitly
HOST=0.0.0.0 npm start
```

### Step 4: Access from Mobile Device
1. Make sure your mobile device is on the same WiFi network as your computer
2. Open your mobile browser
3. Navigate to: `http://YOUR_COMPUTER_IP:3000`
   - Replace `YOUR_COMPUTER_IP` with the IP address you found in Step 1
   - Example: `http://192.168.1.100:3000`

### Step 5: Verify Connection
The app will automatically:
- Detect that you're accessing via IP (not localhost)
- Connect to the backend using the same IP address
- Show connection status in the browser console

## Troubleshooting

### If Still Not Connecting:

1. **Check Firewall**: Make sure your computer's firewall allows connections on ports 3000 and 5001

2. **Check Network**: Ensure both devices are on the same WiFi network

3. **Try Different Ports**: If ports are blocked, you can change them:
   - Backend: Edit `backend/app.py` line 1961, change `port=5001` to `port=5002`
   - Frontend: Create `.env` file with `PORT=3001`

4. **Check Backend Logs**: Look at the backend console for connection attempts

5. **Mobile Browser Console**: Open developer tools on mobile browser to see connection errors

### Alternative: Use ngrok (Advanced)
If you can't get local network access working:

1. Install ngrok: `npm install -g ngrok`
2. Expose backend: `ngrok http 5001`
3. Use the ngrok URL in your mobile browser

## Automatic Detection
The app now automatically detects the correct URL:
- Desktop (localhost): Uses `http://localhost:5001`
- Mobile (IP access): Uses `http://YOUR_IP:5001`
- Production: Uses the same host as the frontend

No manual configuration needed!
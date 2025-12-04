import { io } from "socket.io-client";

// Dynamic URL detection for mobile and desktop compatibility
const getSocketURL = () => {
  // Check if we're in development mode
  if (process.env.NODE_ENV === 'development') {
    // For mobile devices, we need to use the computer's IP address instead of localhost
    // This will be detected automatically or can be set manually
    const hostname = window.location.hostname;
    
    // If accessing via IP address (mobile), use that IP
    if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
      return `http://${hostname}:5001`;
    }
    
    // For desktop localhost access
    return `http://localhost:5001`;
  }
  
  // Production mode - use the same host
  return `${window.location.protocol}//${window.location.hostname}:5001`;
};

export const SOCKET_URL = getSocketURL();

// Debug logging for mobile connectivity
console.log('Socket URL:', SOCKET_URL);
console.log('User Agent:', navigator.userAgent);
console.log('Location:', window.location.href);

// Enhanced socket configuration for maximum stability and mobile compatibility
export const socket = io(SOCKET_URL, {
  transports: ["polling", "websocket"],  // Allow both polling and websocket
  upgrade: true,  // Allow upgrading to websocket for better performance
  reconnection: true,  // Enable automatic reconnection for stability
  reconnectionAttempts: 15,  // Increased attempts for mobile networks
  reconnectionDelay: 2000,  // Increased delay for mobile networks
  reconnectionDelayMax: 15000,  // Increased max delay for mobile networks
  randomizationFactor: 0.5,  // Add randomization to prevent thundering herd
  timeout: 30000,  // Increased timeout for mobile networks
  autoConnect: true,
  forceNew: false,  // Ensure singleton behavior
  multiplex: true,  // Enable multiplexing for better performance
  pingTimeout: 90000,  // Increased ping timeout for mobile networks
  pingInterval: 30000,  // Increased ping interval for mobile networks
  closeOnBeforeunload: false,  // Don't close on beforeunload to maintain connection
  withCredentials: false,  // Disable credentials for local development
  // Mobile-specific optimizations
  rememberUpgrade: true,  // Remember transport upgrade for better mobile performance
  path: '/socket.io/',  // Explicit path for better mobile compatibility
  extraHeaders: {
    'Connection': 'keep-alive',
    'Cache-Control': 'no-cache'
  }
});

// Connection state tracking
let connectionState = {
  isConnected: false,
  reconnectAttempts: 0,
  lastConnected: null,
  connectionErrors: []
};

// Enhanced connection error handling
socket.on('connect', () => {
  console.log('Socket.IO connected successfully');
  connectionState.isConnected = true;
  connectionState.reconnectAttempts = 0;
  connectionState.lastConnected = new Date();
  connectionState.connectionErrors = [];
  
  // Emit connection success event for components to listen to
  socket.emit('client_connected', { timestamp: new Date().toISOString() });
});

socket.on('disconnect', (reason) => {
  console.log('Socket.IO disconnected:', reason);
  connectionState.isConnected = false;
  
  // Don't show alerts for intentional disconnections
  if (reason === 'io client disconnect' || reason === 'io server disconnect') {
    console.log('Disconnection was intentional');
    return;
  }
  
  // Show user-friendly message for unexpected disconnections
  if (typeof window !== 'undefined' && window.alert) {
    window.alert(`Connection lost: ${reason}. Attempting to reconnect...`);
  }
});

socket.on('connect_error', (error) => {
  console.error('Socket.IO connection error:', error);
  connectionState.connectionErrors.push({
    error: error.message,
    timestamp: new Date().toISOString()
  });
  
  // Only show alert on first few connection errors to avoid spam
  if (connectionState.connectionErrors.length <= 3) {
    if (typeof window !== 'undefined' && window.alert) {
      window.alert(`Failed to connect to server at ${SOCKET_URL}. Please ensure the backend server is running on port 5001.\n\nError: ${error.message}`);
    }
  }
});

socket.on('connect_timeout', () => {
  console.error('Socket.IO connection timeout');
  connectionState.connectionErrors.push({
    error: 'Connection timeout',
    timestamp: new Date().toISOString()
  });
  
  if (typeof window !== 'undefined' && window.alert) {
    window.alert('Connection to server timed out. Please check your network connection and ensure the server is running.');
  }
});

socket.on('reconnect', (attemptNumber) => {
  console.log(`Socket.IO reconnected after ${attemptNumber} attempts`);
  connectionState.isConnected = true;
  connectionState.reconnectAttempts = attemptNumber;
  connectionState.lastConnected = new Date();
  
  // Show success message for reconnection
  if (typeof window !== 'undefined' && window.alert) {
    window.alert(`Successfully reconnected to server after ${attemptNumber} attempts!`);
  }
});

socket.on('reconnect_attempt', (attemptNumber) => {
  console.log(`Socket.IO reconnection attempt ${attemptNumber}`);
  connectionState.reconnectAttempts = attemptNumber;
});

socket.on('reconnect_failed', () => {
  console.error('Socket.IO reconnection failed after all attempts');
  connectionState.isConnected = false;
  
  if (typeof window !== 'undefined' && window.alert) {
    window.alert('Unable to reconnect to server after multiple attempts. Please refresh the page or check server status.');
  }
});

// Add heartbeat mechanism to keep connection alive
let heartbeatInterval;
socket.on('connect', () => {
  // Start heartbeat when connected
  if (heartbeatInterval) {
    clearInterval(heartbeatInterval);
  }
  
  heartbeatInterval = setInterval(() => {
    if (socket.connected) {
      socket.emit('ping', { timestamp: new Date().toISOString() });
    }
  }, 30000); // Send ping every 30 seconds
});

socket.on('disconnect', () => {
  // Stop heartbeat when disconnected
  if (heartbeatInterval) {
    clearInterval(heartbeatInterval);
    heartbeatInterval = null;
  }
});

// Handle pong responses
socket.on('pong', (data) => {
  console.log('Received pong from server:', data);
});

// Enhanced error handling for socket events
socket.on('error', (error) => {
  console.error('Socket.IO error:', error);
  connectionState.connectionErrors.push({
    error: error.message || 'Unknown error',
    timestamp: new Date().toISOString()
  });
});

// Connection health check function
export const checkConnectionHealth = () => {
  return {
    isConnected: socket.connected,
    connectionState: connectionState,
    serverUrl: SOCKET_URL,
    lastPing: socket.id ? new Date().toISOString() : null
  };
};

// Manual reconnection function
export const forceReconnect = () => {
  console.log('Forcing socket reconnection...');
  socket.disconnect();
  setTimeout(() => {
    socket.connect();
  }, 1000);
};

// Mobile-specific connection test
export const testMobileConnection = async () => {
  try {
    console.log('Testing mobile connection to:', SOCKET_URL);
    const response = await fetch(`${SOCKET_URL}/test`, {
      method: 'GET',
      timeout: 10000
    });
    const data = await response.json();
    console.log('Mobile connection test successful:', data);
    return { success: true, data };
  } catch (error) {
    console.error('Mobile connection test failed:', error);
    return { success: false, error: error.message };
  }
};

// Gracefully close only on full page unload
if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", () => {
    try { 
      if (heartbeatInterval) {
        clearInterval(heartbeatInterval);
      }
      socket.disconnect(); 
    } catch (_) {}
  });
  
  // Handle page visibility changes to maintain connection
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      // Page is hidden, reduce ping frequency
      if (heartbeatInterval) {
        clearInterval(heartbeatInterval);
        heartbeatInterval = setInterval(() => {
          if (socket.connected) {
            socket.emit('ping', { timestamp: new Date().toISOString() });
          }
        }, 60000); // Send ping every 60 seconds when page is hidden
      }
    } else {
      // Page is visible, resume normal ping frequency
      if (heartbeatInterval) {
        clearInterval(heartbeatInterval);
        heartbeatInterval = setInterval(() => {
          if (socket.connected) {
            socket.emit('ping', { timestamp: new Date().toISOString() });
          }
        }, 30000); // Send ping every 30 seconds when page is visible
      }
    }
  });
}



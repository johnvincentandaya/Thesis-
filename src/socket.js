import { io } from "socket.io-client";

// Explicitly connect to localhost:5001 for backend server
export const SOCKET_URL = `http://localhost:5001`;

// Enhanced socket configuration for maximum stability
export const socket = io(SOCKET_URL, {
  transports: ["polling", "websocket"],  // Allow both polling and websocket
  upgrade: true,  // Allow upgrading to websocket for better performance
  reconnection: true,  // Enable automatic reconnection for stability
  reconnectionAttempts: 10,  // Try to reconnect 10 times for better resilience
  reconnectionDelay: 1000,  // Wait 1 second between attempts
  reconnectionDelayMax: 10000,  // Max 10 seconds between attempts
  randomizationFactor: 0.5,  // Add randomization to prevent thundering herd
  timeout: 20000,  // Reduced timeout for faster failure detection
  autoConnect: true,
  forceNew: false,  // Ensure singleton behavior
  multiplex: true,  // Enable multiplexing for better performance
  pingTimeout: 60000,  // 60 seconds ping timeout
  pingInterval: 25000,  // Ping every 25 seconds to keep connection alive
  closeOnBeforeunload: false,  // Don't close on beforeunload to maintain connection
  withCredentials: false,  // Disable credentials for local development
  extraHeaders: {
    'Connection': 'keep-alive'
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
  console.log('✅ Socket.IO connected successfully');
  connectionState.isConnected = true;
  connectionState.reconnectAttempts = 0;
  connectionState.lastConnected = new Date();
  connectionState.connectionErrors = [];
  
  // Emit connection success event for components to listen to
  socket.emit('client_connected', { timestamp: new Date().toISOString() });
});

socket.on('disconnect', (reason) => {
  console.log('❌ Socket.IO disconnected:', reason);
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
  console.error('❌ Socket.IO connection error:', error);
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
  console.error('⏰ Socket.IO connection timeout');
  connectionState.connectionErrors.push({
    error: 'Connection timeout',
    timestamp: new Date().toISOString()
  });
  
  if (typeof window !== 'undefined' && window.alert) {
    window.alert('Connection to server timed out. Please check your network connection and ensure the server is running.');
  }
});

socket.on('reconnect', (attemptNumber) => {
  console.log(`🔄 Socket.IO reconnected after ${attemptNumber} attempts`);
  connectionState.isConnected = true;
  connectionState.reconnectAttempts = attemptNumber;
  connectionState.lastConnected = new Date();
  
  // Show success message for reconnection
  if (typeof window !== 'undefined' && window.alert) {
    window.alert(`Successfully reconnected to server after ${attemptNumber} attempts!`);
  }
});

socket.on('reconnect_attempt', (attemptNumber) => {
  console.log(`🔄 Socket.IO reconnection attempt ${attemptNumber}`);
  connectionState.reconnectAttempts = attemptNumber;
});

socket.on('reconnect_failed', () => {
  console.error('❌ Socket.IO reconnection failed after all attempts');
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
  console.log('🏓 Received pong from server:', data);
});

// Enhanced error handling for socket events
socket.on('error', (error) => {
  console.error('❌ Socket.IO error:', error);
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
  console.log('🔄 Forcing socket reconnection...');
  socket.disconnect();
  setTimeout(() => {
    socket.connect();
  }, 1000);
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



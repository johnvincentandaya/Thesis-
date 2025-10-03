import { io } from "socket.io-client";

// Explicitly connect to localhost:5001 for backend server
export const SOCKET_URL = `http://localhost:5001`;

// Singleton socket instance to avoid reconnect/disconnect churn under React StrictMode
export const socket = io(SOCKET_URL, {
  transports: ["polling", "websocket"],  // Allow both polling and websocket
  upgrade: true,  // Allow upgrading to websocket for better performance
  reconnection: false,  // Disable automatic reconnection
  timeout: 60000,
  autoConnect: true,
  forceNew: false  // Ensure singleton behavior
});

// Add connection error handling
socket.on('connect_error', (error) => {
  console.error('Socket.IO connection error:', error);
  // Show user-friendly error message
  if (typeof window !== 'undefined' && window.alert) {
    window.alert(`Failed to connect to server at ${SOCKET_URL}. Please ensure the backend server is running on port 5001.`);
  }
});

socket.on('connect_timeout', () => {
  console.error('Socket.IO connection timeout');
  if (typeof window !== 'undefined' && window.alert) {
    window.alert('Connection to server timed out. Please check your network connection and ensure the server is running.');
  }
});

socket.on('reconnect_failed', () => {
  console.error('Socket.IO reconnection failed after all attempts');
  if (typeof window !== 'undefined' && window.alert) {
    window.alert('Unable to reconnect to server after multiple attempts. Please refresh the page or check server status.');
  }
});

// Gracefully close only on full page unload
if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", () => {
    try { socket.disconnect(); } catch (_) {}
  });
}



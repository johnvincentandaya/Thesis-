import { io } from "socket.io-client";

// Resolve server by current hostname to avoid localhost vs 127.0.0.1 mismatches
const host = typeof window !== "undefined" ? window.location.hostname : "127.0.0.1";
export const SOCKET_URL = `http://${host}:5001`;

// Singleton socket instance to avoid reconnect/disconnect churn under React StrictMode
export const socket = io(SOCKET_URL, {
  transports: ["polling"],
  upgrade: false,
  reconnection: true,
  reconnectionAttempts: Infinity,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
  timeout: 60000,
  autoConnect: true
});

// Gracefully close only on full page unload
if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", () => {
    try { socket.disconnect(); } catch (_) {}
  });
}



import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import 'antd/dist/reset.css';
import { UploadProvider } from "./context/UploadContext"; 

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <UploadProvider>  {/* ✅ Wrap App with UploadProvider */}
      <App />
    </UploadProvider>
  </React.StrictMode>
);
reportWebVitals();



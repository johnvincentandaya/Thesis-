import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import UploadDataset from './pages/Models';
import Training from './pages/Training';
import Visualization from './pages/Visualization';
import Instructions from './pages/Instructions';
import Assessment from './pages/Assessment';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/models" element={<UploadDataset />} />
        <Route path="/training" element={<Training />} />
        <Route path="/visualization" element={<Visualization />} />
        <Route path="/instructions" element={<Instructions />} />
        <Route path="/assessment" element={<Assessment />} />
      </Routes>
    </Router>
  );
}

export default App;

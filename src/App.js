import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import UploadDataset from './pages/UploadDataset';
import Training from './pages/Training';
import Evaluation from './pages/Evaluation';
import Visualization from './pages/Visualization';
import Instructions from './pages/Instructions';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/upload" element={<UploadDataset />} />
        <Route path="/training" element={<Training />} />
        <Route path="/evaluation" element={<Evaluation />} />
        <Route path="/visualization" element={<Visualization />} />
        <Route path="/instructions" element={<Instructions />} />
      </Routes>
    </Router>
  );
}

export default App;

# Knowledge Distillation and Pruning Simulator

An interactive educational tool for understanding neural network compression techniques including Knowledge Distillation and Model Pruning.

## Features

### 🎯 Core Functionality
- **Interactive Model Training**: Train and compress neural networks using Knowledge Distillation and Pruning
- **Real-time Visualization**: 3D visualization of neural network compression process
- **Multiple Model Support**: DistilBERT, T5-small, MobileNetV2, and ResNet-18
- **Educational Assessment**: Comprehensive quiz system to test understanding

### 🚀 Recent Improvements

#### Professional UI/UX
- **Clean, Modern Design**: Professional interface without emojis
- **Responsive Layout**: Fully mobile-friendly design
- **Consistent Styling**: Unified color scheme and typography
- **Accessibility**: WCAG compliant with keyboard navigation support

#### Enhanced Training Experience
- **Real Evaluation Results**: Displays actual backend metrics instead of placeholders
- **Persistent Results**: Training results persist across page navigation
- **Smart Navigation**: Next/Previous buttons for browsing evaluation results
- **Training State Management**: Proper button states (Start, Cancel, Train Another Model)

#### Seamless User Flow
- **Auto-Selection**: Models page automatically selects model when navigating to Training
- **Free Navigation**: Users can navigate between pages without interrupting training
- **Back to Training**: Easy navigation from Visualization back to Training page

#### Interactive Visualization
- **Clickable Components**: Click on neural network nodes for educational explanations
- **Educational Content**: Detailed explanations of each component's role
- **Steady Simulation**: Non-chaotic, educational 3D visualization
- **Mobile Optimized**: Touch-friendly 3D controls for mobile devices

#### Mobile Accessibility
- **Responsive Design**: Optimized for all screen sizes
- **Touch Controls**: Full touch support for 3D visualization
- **Mobile-First**: Designed with mobile users in mind
- **Performance Optimized**: Fast loading on mobile networks

## Technology Stack

### Frontend
- **React 18**: Modern React with hooks and functional components
- **React Router**: Client-side routing
- **Ant Design**: Professional UI component library
- **Bootstrap**: Responsive CSS framework
- **Three.js**: 3D visualization with React Three Fiber
- **Socket.IO Client**: Real-time communication

### Backend
- **Flask**: Python web framework
- **Socket.IO**: Real-time bidirectional communication
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library
- **scikit-learn**: Machine learning utilities

## Getting Started

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd kd-pruning-simulator
   ```

2. **Install frontend dependencies**
   ```bash
   npm install
   ```

3. **Install backend dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the backend server**
   ```bash
   cd backend
   python app.py
   ```
   The backend will run on `http://localhost:5001`

2. **Start the frontend development server**
   ```bash
   npm start
   ```
   The frontend will run on `http://localhost:3000`

3. **Open your browser**
   Navigate to `http://localhost:3000` to access the application

## Usage Guide

### 1. Explore Models
- Visit the **Models** page to see available neural network models
- Click on any model to view detailed information
- Click **Start Training** to begin the compression process

### 2. Train Models
- Select a model from the dropdown (or use auto-selection from Models page)
- Click **Start Training** to begin Knowledge Distillation and Pruning
- Monitor real-time progress and metrics
- Use **Cancel Training** to stop if needed
- Click **Train Another Model** after completion

### 3. Visualize Results
- After training, proceed to the **Visualization** page
- Watch the 3D neural network compression process
- Click on nodes for educational explanations
- Use **Back to Training** to return to training results

### 4. Test Knowledge
- Take the **Assessment** quiz to test your understanding
- Review detailed explanations for each answer
- Track your progress and learning outcomes

## Mobile Usage

The application is fully optimized for mobile devices:

- **Touch Controls**: Use touch gestures to interact with 3D visualization
- **Responsive Layout**: All components adapt to mobile screen sizes
- **Mobile Navigation**: Touch-friendly navigation and controls
- **Performance**: Optimized for mobile networks and devices

See [MOBILE_GUIDE.md](MOBILE_GUIDE.md) for detailed mobile accessibility information.

## Architecture

### Frontend Architecture
```
src/
├── components/          # Reusable UI components
├── pages/              # Main application pages
│   ├── Home.js         # Landing page
│   ├── Models.js       # Model selection and information
│   ├── Training.js     # Training interface
│   ├── Visualization.js # 3D visualization
│   └── Assessment.js   # Knowledge assessment
├── App.js              # Main application component
└── App.css             # Global styles and mobile responsiveness
```

### Backend Architecture
```
backend/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
└── uploads/           # File upload directory
```

## Key Features Explained

### Knowledge Distillation
- **Teacher-Student Learning**: Large teacher model transfers knowledge to smaller student model
- **Soft Targets**: Student learns from teacher's probability distributions
- **Temperature Scaling**: Controls the softness of knowledge transfer
- **Efficiency Gains**: Significant size reduction with minimal accuracy loss

### Model Pruning
- **Weight Removal**: Eliminates redundant or less important connections
- **Sparsity Introduction**: Creates sparse neural networks
- **Performance Trade-offs**: Balances model size vs. accuracy
- **Structured Pruning**: Removes entire neurons, filters, or layers

### Real-time Visualization
- **3D Neural Networks**: Interactive 3D representation of network structure
- **Compression Process**: Visual demonstration of pruning effects
- **Educational Explanations**: Clickable components with detailed information
- **Mobile Support**: Touch-optimized 3D controls

## Development

### Available Scripts

- `npm start`: Start development server
- `npm test`: Run test suite
- `npm run build`: Build for production
- `npm run eject`: Eject from Create React App (not recommended)

### Code Style
- ESLint configuration for consistent code style
- Prettier for code formatting
- Component-based architecture
- Functional components with hooks

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Hugging Face**: For the Transformers library and pre-trained models
- **PyTorch**: For the deep learning framework
- **Three.js**: For 3D visualization capabilities
- **Ant Design**: For the professional UI components
- **React Community**: For the excellent React ecosystem

## Support

For support, please open an issue in the GitHub repository or contact the development team.

## Roadmap

### Upcoming Features
- [ ] Additional model architectures
- [ ] Advanced pruning techniques
- [ ] Performance benchmarking tools
- [ ] Export functionality for trained models
- [ ] Collaborative features
- [ ] Advanced visualization options

### Mobile Enhancements
- [ ] Progressive Web App (PWA) support
- [ ] Offline functionality
- [ ] Push notifications
- [ ] Native app development

---

**Built with ❤️ for the AI/ML education community**
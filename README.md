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

## Installation and Setup Guide

This guide provides step-by-step instructions to install, configure, and run the Knowledge Distillation and Pruning Simulator project.

### Prerequisites

Before starting, ensure you have the following installed on your system:

1. **Node.js and npm**
   - Version: Node.js 16 or higher
   - Download: [https://nodejs.org/](https://nodejs.org/)
   - Verify installation:
     ```bash
     node --version
     npm --version
     ```

2. **Python**
   - Version: Python 3.8 or higher
   - Download: [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Verify installation:
     ```bash
     python --version
     ```
   - Note: On some systems, use `python3` instead of `python`

3. **pip (Python Package Manager)**
   - Usually comes with Python installation
   - Verify installation:
     ```bash
     pip --version
     ```
   - Note: On some systems, use `pip3` instead of `pip`

4. **Git (Optional)**
   - Only needed if cloning from a repository
   - Download: [https://git-scm.com/downloads](https://git-scm.com/downloads)

### Step-by-Step Installation

#### Step 1: Access the Project Folder

1. Navigate to the project folder location on your computer
2. Open a terminal/command prompt in the project root directory
   - **Windows**: Right-click in the folder → "Open PowerShell here" or "Open Command Prompt here"
   - **Mac/Linux**: Open Terminal and use `cd` to navigate to the project folder

#### Step 2: Install Frontend Dependencies

1. Make sure you're in the project root directory (where `package.json` is located)
2. Run the following command:
   ```bash
   npm install
   ```
3. Wait for the installation to complete (this may take 2-5 minutes)
4. You should see a message indicating successful installation

**Troubleshooting:**
- If you get permission errors, try: `sudo npm install` (Mac/Linux) or run as Administrator (Windows)
- If npm is not found, ensure Node.js is properly installed and added to your system PATH

#### Step 3: Install Backend Dependencies

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   - On some systems, use: `pip3 install -r requirements.txt`
3. Wait for the installation to complete (this may take 5-10 minutes as it downloads PyTorch and other large packages)
4. You should see a message indicating successful installation

**Troubleshooting:**
- If you get permission errors, try: `pip install --user -r requirements.txt`
- If pip is not found, ensure Python is properly installed and added to your system PATH
- On Windows, you may need to use: `python -m pip install -r requirements.txt`

#### Step 4: Verify Installation

1. Check that frontend dependencies are installed:
   ```bash
   cd ..  # Go back to project root
   ls node_modules  # Should show many folders (Mac/Linux)
   dir node_modules  # Should show many folders (Windows)
   ```

2. Check that backend dependencies are installed:
   ```bash
   cd backend
   pip list  # Should show Flask, PyTorch, transformers, etc.
   ```

### Running the Application

The application consists of two parts that need to run simultaneously:
1. **Backend Server** (Flask/Python) - Handles model training and processing
2. **Frontend Server** (React/Node.js) - Provides the user interface

#### Option A: Using Two Terminal Windows (Recommended)

**Terminal 1 - Backend Server:**
1. Open a terminal/command prompt
2. Navigate to the backend directory:
   ```bash
   cd backend
   ```
3. Start the backend server:
   ```bash
   python app.py
   ```
   - On some systems, use: `python3 app.py`
4. Wait for the message: "Server is running on http://localhost:5001" or similar
5. **Keep this terminal window open** - the backend must stay running

**Terminal 2 - Frontend Server:**
1. Open a **new** terminal/command prompt window
2. Navigate to the project root directory (where `package.json` is located)
3. Start the frontend server:
   ```bash
   npm start
   ```
4. Wait for the browser to automatically open, or manually navigate to `http://localhost:3000`
5. **Keep this terminal window open** - the frontend must stay running

#### Option B: Using Background Processes

**Windows (PowerShell):**
```powershell
# Start backend in background
Start-Process python -ArgumentList "backend/app.py" -WindowStyle Hidden

# Start frontend
npm start
```

**Mac/Linux:**
```bash
# Start backend in background
cd backend && python app.py &
cd ..

# Start frontend
npm start
```

### Accessing the Application

1. Once both servers are running, open your web browser
2. Navigate to: `http://localhost:3000`
3. You should see the Knowledge Distillation and Pruning Simulator homepage

### Verifying Everything Works

1. **Check Backend Connection:**
   - On the Training page, you should see "Server Status: Connected" (green indicator)
   - If it shows "Error" or "Checking", the backend may not be running properly

2. **Test Model Selection:**
   - Go to the Models page
   - You should see 4 model cards (DistilBERT, T5-small, MobileNetV2, ResNet-18)

3. **Test Training:**
   - Go to the Training page
   - Select a model from the dropdown
   - Click "Start Training" to test the system

### Common Issues and Solutions

**Issue: "Cannot connect to server" or "Server Status: Error"**
- **Solution**: Make sure the backend server is running in a separate terminal
- Check that port 5001 is not being used by another application
- Verify the backend started successfully (check Terminal 1 for error messages)

**Issue: "npm start" fails or shows errors**
- **Solution**: Make sure you're in the project root directory (where `package.json` is located)
- Try deleting `node_modules` folder and `package-lock.json`, then run `npm install` again

**Issue: "Module not found" errors in Python**
- **Solution**: Make sure you installed all requirements: `pip install -r requirements.txt`
- Verify you're using the correct Python version (3.8+)
- Try creating a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # Mac/Linux
  venv\Scripts\activate  # Windows
  pip install -r requirements.txt
  ```

**Issue: Port already in use**
- **Solution**: 
  - Backend (port 5001): Change the port in `backend/app.py` or close the application using port 5001
  - Frontend (port 3000): The terminal will ask if you want to use a different port - type "Y" and press Enter

**Issue: Training fails or models don't load**
- **Solution**: 
  - Ensure you have a stable internet connection (models are downloaded from Hugging Face)
  - Check that you have sufficient disk space (models can be several GB)
  - Verify PyTorch is properly installed: `python -c "import torch; print(torch.__version__)"`

### Stopping the Application

1. **Stop Frontend**: In the terminal running `npm start`, press `Ctrl + C`
2. **Stop Backend**: In the terminal running `python app.py`, press `Ctrl + C`
3. Close both terminal windows

### Project Structure

```
Project Root/
├── src/                    # Frontend React application
│   ├── pages/             # Main application pages
│   ├── components/        # Reusable components
│   └── ...
├── backend/               # Backend Flask application
│   ├── app.py            # Main Flask server
│   ├── requirements.txt   # Python dependencies
│   └── ...
├── public/               # Static files
├── package.json          # Frontend dependencies
└── README.md            # This file
```

### Next Steps

Once the application is running:
1. Read the **Instructions** page for a detailed guide on using the simulator
2. Explore the **Models** page to see available neural network models
3. Try training a model on the **Training** page
4. View results in the **Visualization** page (available after training)
5. Test your knowledge with the **Assessment** quiz

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


---
import { useState } from 'react'
import './App.css'
import FileUpload from './components/FileUpload'
import ManualInput from './components/ManualInput'
import ImageUpload from './components/ImageUpload'
import Results from './components/Results'

function App() {
    const [prediction, setPrediction] = useState(null)
    const [loading, setLoading] = useState(false)
    const [activeTab, setActiveTab] = useState('image') // Default to image for new feature

    const handlePrediction = async (data) => {
        setLoading(true)
        setPrediction(null)

        try {
            let endpoint = 'http://localhost:8000/predict'
            let body
            let headers = {}

            // Check if data is a File object (Image) or Array (Sensor Data)
            if (data instanceof File) {
                endpoint = 'http://localhost:8000/predict_image'
                const formData = new FormData()
                formData.append('file', data)
                body = formData
                // Content-Type header is auto-set by browser for FormData
            } else {
                headers = { 'Content-Type': 'application/json' }
                body = JSON.stringify({ data: data })
            }

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: headers,
                body: body
            })

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}))
                throw new Error(errorData.detail || 'Prediction failed')
            }

            const result = await response.json()
            setPrediction(result)
        } catch (error) {
            console.error('Error:', error)
            alert(`Error: ${error.message}\n\nMake sure the backend server is running on port 8000.`)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="App">
            <header className="app-header">
                <h1>Movement AI</h1>
                <p>Advanced Activity Recognition using Deep Learning & Computer Vision</p>
            </header>

            <main className="main-content">
                <div className="container">
                    {/* Tab Navigation */}
                    <div className="tabs">
                        <button
                            className={`tab ${activeTab === 'image' ? 'active' : ''}`}
                            onClick={() => { setActiveTab('image'); setPrediction(null); }}
                        >
                            <span className="tab-label">Visual Input</span>
                        </button>
                        <button
                            className={`tab ${activeTab === 'file' ? 'active' : ''}`}
                            onClick={() => { setActiveTab('file'); setPrediction(null); }}
                        >
                            <span className="tab-label">Sensor Data</span>
                        </button>
                        <button
                            className={`tab ${activeTab === 'manual' ? 'active' : ''}`}
                            onClick={() => { setActiveTab('manual'); setPrediction(null); }}
                        >
                            <span className="tab-label">Manual Lab</span>
                        </button>
                    </div>

                    {/* Tab Content */}
                    <div className="tab-content">
                        {activeTab === 'image' ? (
                            <ImageUpload onPredict={handlePrediction} loading={loading} />
                        ) : activeTab === 'file' ? (
                            <FileUpload onPredict={handlePrediction} loading={loading} />
                        ) : (
                            <ManualInput onPredict={handlePrediction} loading={loading} />
                        )}
                    </div>

                    {/* Results */}
                    {prediction && <Results prediction={prediction} />}
                </div>
            </main>

            <footer className="app-footer">
                <p>Built with PyTorch LSTM + CLIP &bull; UCI HAR Research Dataset</p>
            </footer>
        </div>
    )
}

export default App

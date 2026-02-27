import { useState } from 'react'
import './FileUpload.css'

function FileUpload({ onPredict, loading }) {
    const [file, setFile] = useState(null)
    const [dragActive, setDragActive] = useState(false)

    const handleDrag = (e) => {
        e.preventDefault()
        e.stopPropagation()
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true)
        } else if (e.type === "dragleave") {
            setDragActive(false)
        }
    }

    const handleDrop = (e) => {
        e.preventDefault()
        e.stopPropagation()
        setDragActive(false)

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0])
        }
    }

    const handleChange = (e) => {
        e.preventDefault()
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0])
        }
    }

    const handleSubmit = async (e) => {
        e.preventDefault()

        if (!file) {
            alert('Please select a file first!')
            return
        }

        try {
            const text = await file.text()

            // Parse CSV/TXT file
            // Expected format: 128 rows × 9 columns (comma or space separated)
            const lines = text.trim().split('\n')
            const data = lines.map(line => {
                const values = line.trim().split(/[,\s]+/).map(Number)
                if (values.length !== 9) {
                    throw new Error(`Invalid data format. Expected 9 values per row, got ${values.length}`)
                }
                return values
            })

            if (data.length !== 128) {
                throw new Error(`Invalid data format. Expected 128 timesteps, got ${data.length}`)
            }

            // Call prediction
            await onPredict(data)
        } catch (error) {
            console.error('Error processing file:', error)
            alert(`Error: ${error.message}`)
        }
    }

    return (
        <div className="file-upload glass-card">
            <form onSubmit={handleSubmit}>
                <div
                    className={`upload-area ${dragActive ? 'drag-active' : ''}`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                >
                    <input
                        type="file"
                        id="file-input"
                        accept=".txt,.csv"
                        onChange={handleChange}
                        style={{ display: 'none' }}
                    />

                    <label htmlFor="file-input" className="upload-label">
                        {file ? (
                            <div className="file-info">
                                <p className="file-name">{file.name}</p>
                                <p className="file-size">{(file.size / 1024).toFixed(2)} KB</p>
                            </div>
                        ) : (
                            <>
                                <p className="upload-text">Drag sensor data file here</p>
                                <p className="upload-hint">Supported formats: .txt, .csv</p>
                            </>
                        )}
                    </label>
                </div>

                <div className="file-format-info">
                    <h3>Data Format Requirements</h3>
                    <p>Sensor sequence data must follow these specifications:</p>
                    <ul>
                        <li><strong>128 timesteps</strong> (sequentially ordered)</li>
                        <li><strong>9 features</strong> (Acc X/Y/Z, Gyro X/Y/Z, Total Acc X/Y/Z)</li>
                        <li>Delimited by commas or whitespace</li>
                    </ul>
                </div>

                <button
                    type="submit"
                    className="primary-btn"
                    disabled={!file || loading}
                    style={{ width: '100%', marginTop: '1.5rem' }}
                >
                    {loading ? (
                        <>
                            <span className="loading-spinner"></span>
                            Computing...
                        </>
                    ) : (
                        "Analyze Sensor Sequence"
                    )}
                </button>
            </form>
        </div>
    )
}

export default FileUpload

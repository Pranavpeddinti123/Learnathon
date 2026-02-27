import { useState } from 'react'
import './FileUpload.css'

function ImageUpload({ onPredict, loading }) {
    const [file, setFile] = useState(null)
    const [preview, setPreview] = useState(null)
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
            handleFileSelection(e.dataTransfer.files[0])
        }
    }

    const handleChange = (e) => {
        e.preventDefault()
        if (e.target.files && e.target.files[0]) {
            handleFileSelection(e.target.files[0])
        }
    }

    const handleFileSelection = (selectedFile) => {
        // Validate file type
        if (!selectedFile.type.startsWith('image/')) {
            alert('Please select an image file (JPG, PNG)')
            return
        }

        setFile(selectedFile)

        // Create preview URL
        const objectUrl = URL.createObjectURL(selectedFile)
        setPreview(objectUrl)
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        if (!file) {
            alert('Please select an image first!')
            return
        }

        // We pass the actual file object for prediction
        onPredict(file)
    }

    return (
        <div className="file-upload glass-card">
            <div
                className={`upload-area ${dragActive ? 'drag-active' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                style={{ minHeight: '300px', borderRadius: '16px' }}
            >
                <input
                    type="file"
                    id="image-input"
                    accept="image/*"
                    onChange={handleChange}
                    style={{ display: 'none' }}
                />

                <label htmlFor="image-input" className="upload-label" style={{ display: 'block', width: '100%', height: '100%', cursor: 'pointer' }}>
                    {preview ? (
                        <div className="image-preview-container" style={{ textAlign: 'center' }}>
                            <img
                                src={preview}
                                alt="Preview"
                                style={{
                                    maxHeight: '300px',
                                    maxWidth: '100%',
                                    borderRadius: '12px',
                                    boxShadow: '0 12px 24px rgba(0,0,0,0.3)',
                                    marginBottom: '1rem'
                                }}
                            />
                            <div className="file-info">
                                <p className="file-name">{file.name}</p>
                            </div>
                            <p className="upload-hint" style={{ marginTop: '1rem' }}>Change image</p>
                        </div>
                    ) : (
                        <div style={{ padding: '2rem' }}>
                            <p className="upload-text">Upload motion capture image</p>
                            <p className="upload-hint">JPG, PNG or WEBP formats supported</p>
                        </div>
                    )}
                </label>
            </div>

            <div className="file-format-info" style={{ borderColor: 'var(--primary)', background: 'rgba(79, 70, 229, 0.05)', borderRadius: '12px' }}>
                <h3 style={{ color: '#818cf8', marginBottom: '0.75rem' }}>Computer Vision Analysis</h3>
                <p>The AI model identifies dynamic states including:</p>
                <ul style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', listStyle: 'none', padding: 0 }}>
                    <li>&bull; Walking & Running</li>
                    <li>&bull; Sitting & Standing</li>
                    <li>&bull; Rest States</li>
                    <li>&bull; Elevation Movement</li>
                </ul>
            </div>

            <button
                type="button"
                className="primary-btn"
                disabled={!file || loading}
                onClick={handleSubmit}
                style={{ width: '100%', marginTop: '1.5rem' }}
            >
                {loading ? (
                    <>
                        <span className="loading-spinner"></span>
                        Analyzing Image...
                    </>
                ) : (
                    "Run Vision Inference"
                )}
            </button>
        </div>
    )
}

export default ImageUpload

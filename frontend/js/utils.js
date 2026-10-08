function setText(id, value) { document.getElementById(id).textContent = value; }

function formatMemory(bytes) {
	if (!bytes) return 'Unavailable';
	return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
}

function formatFileSize(bytes) {
	if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
	if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
	return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

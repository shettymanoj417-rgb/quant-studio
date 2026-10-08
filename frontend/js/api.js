const API_BASE = 'http://127.0.0.1:5000/api';

async function requestJson(path, options = {}) {
	const response = await fetch(`${API_BASE}${path}`, options);
	const data = await response.json().catch(() => ({}));
	if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
	return data;
}

function getDeviceInfo() { return requestJson('/device-info'); }

function quantizeModel(file, quantization, deviceMap) {
	const body = new FormData();
	body.append('model', file);
	body.append('quantization', quantization);
	body.append('device_map', deviceMap);
	return requestJson('/quantize', { method: 'POST', body });
}

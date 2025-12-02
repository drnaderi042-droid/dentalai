# Advanced Image Editor for Cephalometric Analysis

A professional image annotation tool similar to makesense.ai, built specifically for cephalometric analysis.

## Features

### 🎨 Annotation Tools
- **Point Tool** - Mark specific anatomical landmarks
- **Rectangle Tool** - Draw bounding boxes
- **Polygon Tool** - Create closed shapes for complex anatomical structures
- **Polyline Tool** - Draw freeform lines for tracing anatomical features
- **Landmark Tool** - Specialized point markers with labels

### ⚡ Advanced Features
- **Undo/Redo** - Full history management (Ctrl+Z / Ctrl+Y)
- **Zoom & Pan** - Navigate large images easily
- **Object Management Panel** - View and manage all annotations
- **Visibility Controls** - Toggle annotation visibility
- **Context Menu** - Right-click for quick actions
- **Keyboard Shortcuts** - Fast workflow

### 📤 Export Formats
- **JSON** - Native format with all annotation data
- **CSV** - Spreadsheet-compatible format
- **COCO** - Standard computer vision format

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `P` | Point Tool |
| `R` | Rectangle Tool |
| `G` | Polygon Tool |
| `L` | Polyline Tool |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` or `Ctrl+Shift+Z` | Redo |
| `Delete` or `Backspace` | Delete selected annotation |
| `Esc` | Cancel current drawing |
| `Ctrl+Click` or `Middle Mouse` | Pan mode |
| `Mouse Wheel` | Zoom in/out |

## Usage Example

### Basic Integration

```jsx
import { AdvancedImageEditor } from 'src/sections/orthodontics/cephalometric-analysis/advanced-image-editor';

function CephalometricAnalysisPage() {
  const [imageUrl, setImageUrl] = useState('');
  const [annotations, setAnnotations] = useState([]);

  const handleAnnotationsChange = (newAnnotations) => {
    setAnnotations(newAnnotations);
    // Save to backend or local storage
    console.log('Annotations updated:', newAnnotations);
  };

  return (
    <AdvancedImageEditor
      imageUrl={imageUrl}
      onAnnotationsChange={handleAnnotationsChange}
    />
  );
}
```

### With File Upload

```jsx
import { useState } from 'react';
import { AdvancedImageEditor } from 'src/sections/orthodontics/cephalometric-analysis/advanced-image-editor';
import { Button, Box } from '@mui/material';

function CephalometricPage() {
  const [imageUrl, setImageUrl] = useState(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setImageUrl(url);
    }
  };

  return (
    <Box>
      <input
        type="file"
        accept="image/*"
        onChange={handleImageUpload}
        style={{ display: 'none' }}
        id="image-upload"
      />
      <label htmlFor="image-upload">
        <Button variant="contained" component="span">
          Upload Cephalometric Image
        </Button>
      </label>

      {imageUrl && (
        <AdvancedImageEditor
          imageUrl={imageUrl}
          onAnnotationsChange={(annotations) => {
            console.log('Annotations:', annotations);
          }}
        />
      )}
    </Box>
  );
}
```

## Annotation Data Structure

```javascript
{
  annotations: [
    {
      type: 'point',           // 'point', 'rectangle', 'polygon', 'polyline', 'landmark'
      x: 150,                  // X coordinate (for points/rectangles)
      y: 200,                  // Y coordinate (for points/rectangles)
      width: 50,               // Width (for rectangles)
      height: 30,              // Height (for rectangles)
      points: [{x, y}, ...],   // Array of points (for polygons/polylines)
      color: '#FF0000',        // Annotation color
      thickness: 2,            // Line thickness
      opacity: 1,              // Opacity (0-1)
      radius: 5,               // Point radius
      visible: true,           // Visibility flag
      label: 'Nasion'         // Optional label
    }
  ],
  imageUrl: 'path/to/image.jpg'
}
```

## Integration with Existing Code

To integrate with your existing cephalometric canvas component:

```jsx
// In your cephalometric analysis view
import { AdvancedImageEditor } from './cephalometric-analysis/advanced-image-editor';

// Replace the old CephalometricCanvas with AdvancedImageEditor
<AdvancedImageEditor
  imageUrl={patient.cephalometricImage}
  onAnnotationsChange={handleSaveAnnotations}
/>
```

## Customization

### Custom Landmark Definitions

You can customize the default landmarks by modifying the `DEFAULT_LANDMARKS` constant in the component:

```jsx
const DEFAULT_LANDMARKS = {
  S: { name: 'Sella', color: '#2196F3' },
  N: { name: 'Nasion', color: '#2196F3' },
  // Add your custom landmarks here
};
```

### Styling

The component uses Material-UI theming. You can customize colors and styles through your theme:

```jsx
import { ThemeProvider, createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#YOUR_COLOR',
    },
  },
});

<ThemeProvider theme={theme}>
  <AdvancedImageEditor {...props} />
</ThemeProvider>
```

## API Reference

### Props

| Prop | Type | Required | Description |
|------|------|----------|-------------|
| `imageUrl` | string | Yes | URL or data URI of the cephalometric image |
| `onAnnotationsChange` | function | No | Callback when annotations change `(annotations) => void` |

### Methods

The component handles all interactions internally. Access to annotations is provided through the `onAnnotationsChange` callback.

## Best Practices

1. **Image Loading**: Ensure CORS headers are set correctly on your image server
2. **Performance**: For large images, consider using image compression
3. **Data Persistence**: Save annotations to backend after each change
4. **Validation**: Validate annotation data before saving
5. **User Feedback**: Provide clear instructions for first-time users

## Troubleshooting

### Canvas Tainting Error
If you see "Tainted canvases may not be exported":
- Ensure images are served with proper CORS headers
- Set `crossOrigin="anonymous"` on images (already handled in component)

### Slow Performance
- Reduce image size before annotation
- Limit the number of concurrent annotations
- Use `requestAnimationFrame` for smooth rendering

### Annotations Not Saving
- Check that `onAnnotationsChange` callback is properly connected
- Verify network requests if saving to backend
- Check browser console for errors

## License

Part of the Dental AI project.
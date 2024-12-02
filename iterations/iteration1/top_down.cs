using UnityEngine;
using System.IO;

public class TopDownImageCapture : MonoBehaviour
{
    [Header("Camera Settings")]
    [SerializeField] private Vector3 cameraPosition = new Vector3(0, 10, 0);
    [SerializeField] private float orthographicSize = 5f;
    [SerializeField] private LayerMask captureLayers = -1; // All layers by default
    
    [Header("Output Settings")]
    [SerializeField] private int imageWidth = 1024;
    [SerializeField] private int imageHeight = 1024;
    [SerializeField] private string outputPath = "TopDownCaptures";
    
    private Camera topDownCamera;
    private RenderTexture renderTexture;
    
    void Awake()
    {
        // Create and setup camera
        GameObject cameraObject = new GameObject("TopDownCamera");
        topDownCamera = cameraObject.AddComponent<Camera>();
        
        // Configure camera
        topDownCamera.transform.position = cameraPosition;
        topDownCamera.transform.rotation = Quaternion.Euler(90, 0, 0);
        topDownCamera.orthographic = true;
        topDownCamera.orthographicSize = orthographicSize;
        topDownCamera.cullingMask = captureLayers;
        
        // Create render texture
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        topDownCamera.targetTexture = renderTexture;
    }
    
    public void CaptureImage(string filename = "topdown")
    {
        // Create directory if it doesn't exist
        if (!Directory.Exists(outputPath))
        {
            Directory.CreateDirectory(outputPath);
        }
        
        // Create temporary texture and render the camera view
        RenderTexture.active = renderTexture;
        topDownCamera.Render();
        
        Texture2D screenshot = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        screenshot.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        screenshot.Apply();
        
        // Convert to bytes and save
        byte[] bytes = screenshot.EncodeToPNG();
        string fullPath = Path.Combine(outputPath, $"{filename}_{System.DateTime.Now:yyyyMMdd_HHmmss}.png");
        File.WriteAllBytes(fullPath, bytes);
        
        // Cleanup
        Destroy(screenshot);
        Debug.Log($"Screenshot saved to: {fullPath}");
    }
    
    private void OnDestroy()
    {
        if (renderTexture != null)
        {
            renderTexture.Release();
        }
    }
}

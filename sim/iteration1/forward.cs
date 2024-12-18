using UnityEngine;
using System.IO;

public class ForwardCameraCapture : MonoBehaviour
{
    [Header("Camera Settings")]
    [SerializeField] private Vector3 cameraPosition = new Vector3(0, 2, -5);
    [SerializeField] private Vector3 lookAtPoint = Vector3.zero;
    [SerializeField] private float fieldOfView = 60f;
    [SerializeField] private LayerMask captureLayers = -1; // All layers by default
    
    [Header("Output Settings")]
    [SerializeField] private int imageWidth = 1920;
    [SerializeField] private int imageHeight = 1080;
    [SerializeField] private string outputPath = "ForwardCaptures";
    
    private Camera forwardCamera;
    private RenderTexture renderTexture;
    
    void Awake()
    {
        // Create and setup camera
        GameObject cameraObject = new GameObject("ForwardCamera");
        forwardCamera = cameraObject.AddComponent<Camera>();
        
        // Configure camera
        forwardCamera.transform.position = cameraPosition;
        forwardCamera.transform.LookAt(lookAtPoint);
        forwardCamera.fieldOfView = fieldOfView;
        forwardCamera.cullingMask = captureLayers;
        
        // Create render texture
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        forwardCamera.targetTexture = renderTexture;
        
        // Make camera a child of this GameObject
        cameraObject.transform.SetParent(transform);
    }
    
    public void CaptureImage(string filename = "forward_view")
    {
        // Create directory if it doesn't exist
        if (!Directory.Exists(outputPath))
        {
            Directory.CreateDirectory(outputPath);
        }
        
        // Create temporary texture and render the camera view
        RenderTexture.active = renderTexture;
        forwardCamera.Render();
        
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
    
    // Helper method to update camera position and rotation at runtime
    public void UpdateCameraPosition(Vector3 newPosition, Vector3 newLookAt)
    {
        forwardCamera.transform.position = newPosition;
        forwardCamera.transform.LookAt(newLookAt);
    }
    
    // Helper method to update field of view at runtime
    public void UpdateFieldOfView(float newFOV)
    {
        forwardCamera.fieldOfView = newFOV;
    }
    
    private void OnDestroy()
    {
        if (renderTexture != null)
        {
            renderTexture.Release();
        }
    }
}

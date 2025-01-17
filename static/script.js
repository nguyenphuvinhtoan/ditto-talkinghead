// static/script.js
document.addEventListener("DOMContentLoaded", function () {
  const imageInput = document.getElementById("imageInput");
  const audioInput = document.getElementById("audioInput");
  const processButton = document.getElementById("processButton");
  const imagePreview = document.getElementById("imagePreview");
  const audioPlayer = document.getElementById("audioPlayer");
  const videoPreview = document.getElementById("videoPreview");
  const progressBar = document.querySelector(".progress");

  let imageFile = null;
  let audioFile = null;

  // Image upload handler
  imageInput.addEventListener("change", function (e) {
    imageFile = e.target.files[0];
    if (imageFile) {
      const reader = new FileReader();
      reader.onload = function (e) {
        imagePreview.innerHTML = `<img src="${e.target.result}" style="max-width: 100%;">`;
      };
      reader.readAsDataURL(imageFile);
      checkFiles();
    }
  });

  // Audio upload handler
  audioInput.addEventListener("change", function (e) {
    audioFile = e.target.files[0];
    if (audioFile) {
      const url = URL.createObjectURL(audioFile);
      audioPlayer.src = url;
      audioPlayer.style.display = "block";
      checkFiles();
    }
  });

  // Check if both files are uploaded
  function checkFiles() {
    processButton.disabled = !(imageFile && audioFile);
  }

  // Process button handler
  processButton.addEventListener("click", async function () {
    if (!imageFile || !audioFile) return;

    // Show progress bar
    progressBar.style.display = "block";
    processButton.disabled = true;

    try {
      const formData = new FormData();
      formData.append("image", imageFile);
      formData.append("audio", audioFile);

      const videoContainer = document.getElementById("videoPreview");
      videoContainer.style.display = "block";

      // Create canvas for displaying frames
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      canvas.style.width = "100%";
      videoContainer.parentNode.replaceChild(canvas, videoContainer);

      const response = await fetch("http://localhost:8000/process", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Handle the multipart stream
      const reader = new ReadableStreamDefaultReader(response.body);
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split("\r\n--frame\r\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          if (part.includes("Content-Type: image/jpeg")) {
            const imageBlob = new Blob([part.split("\r\n\r\n")[1]], {
              type: "image/jpeg",
            });
            const img = new Image();
            img.onload = () => {
              canvas.width = img.width;
              canvas.height = img.height;
              ctx.drawImage(img, 0, 0);
            };
            img.src = URL.createObjectURL(imageBlob);
          }
        }
      }
    } catch (error) {
      console.error("Streaming error:", error);
      alert("Error during video streaming");
    } finally {
      progressBar.style.display = "none";
      processButton.disabled = false;
    }
  });
});

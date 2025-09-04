// Classifier page code
if (document.getElementById("uploadBox")) {
  const uploadBox = document.getElementById("uploadBox");
  const fileInput = document.getElementById("fileInput");
  const previewImage = document.getElementById("previewImage");
  const predictBtn = document.getElementById("predictBtn");
  const result = document.getElementById("result");

  // Handle click on upload area
  uploadBox.addEventListener("click", () => fileInput.click());

  // Handle file select
  fileInput.addEventListener("change", handleFile);

  // Drag & Drop
  uploadBox.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadBox.style.background = "#e8ffe8";
  });

  uploadBox.addEventListener("dragleave", () => {
    uploadBox.style.background = "#f9fff6";
  });

  uploadBox.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadBox.style.background = "#f9fff6";
    fileInput.files = e.dataTransfer.files;
    handleFile();
  });

  // Show preview
  function handleFile() {
    const file = fileInput.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        previewImage.style.display = "block";
        previewImage.src = e.target.result;
        predictBtn.disabled = false;
      };
      reader.readAsDataURL(file);
    }
  }

  // Real prediction using backend API
  predictBtn.addEventListener("click", () => {
    const file = fileInput.files[0];
    if (!file) {
      result.innerHTML = "<h2>Please select a file first.</h2>";
      return;
    }
    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        if (data.error) {
          result.innerHTML = `<h2>Error: ${data.error}</h2>`;
        } else {
          const prediction = data.prediction;
          const breedMatch = prediction.match(/^(.+?)\s*\(/);
          const breedName = breedMatch ? breedMatch[1].trim() : prediction;
          result.innerHTML = `<h2>🔮 Prediction: ${prediction}</h2>
          <a href="/breed_detail/${encodeURIComponent(breedName)}" class="btn btn-outline-success mt-3">Read More</a>`;
        }
      })
      .catch((error) => {
        result.innerHTML = `<h2>Error: ${error.message}</h2>`;
      });
  });
}

// Animated counts for dataset page
if (document.querySelector('.dataset-container')) {
  const statCards = document.querySelectorAll('.stat-card p');

  function animateNumber(element, target, prefix = '~ ', suffix = '', duration = 2000) {
    let start = 0;
    const increment = target / (duration / 16); // 60fps
    const timer = setInterval(() => {
      start += increment;
      if (start >= target) {
        start = target;
        clearInterval(timer);
      }
      element.textContent = prefix + Math.floor(start).toLocaleString() + suffix;
    }, 16);
  }

  statCards.forEach(card => {
    const text = card.textContent.trim();
    if (text.includes('~')) {
      const numStr = text.replace('~ ', '').replace(',', '');
      const target = parseInt(numStr);
      if (!isNaN(target)) {
        animateNumber(card, target);
      }
    } else if (text.includes('%')) {
      // For split: 80% / 20%
      const parts = text.split(' / ');
      if (parts.length === 2) {
        const num1 = parseInt(parts[0]);
        const num2 = parseInt(parts[1]);
        let start1 = 0, start2 = 0;
        const increment1 = num1 / (2000 / 16);
        const increment2 = num2 / (2000 / 16);
        const timer = setInterval(() => {
          start1 += increment1;
          start2 += increment2;
          if (start1 >= num1 && start2 >= num2) {
            start1 = num1;
            start2 = num2;
            clearInterval(timer);
          }
          card.textContent = Math.floor(start1) + '% / ' + Math.floor(start2) + '%';
        }, 16);
      }
    }
  });
}

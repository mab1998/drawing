var settings = {
  max_width: 600,
  max_height: 200
}
		setTimeout(initCropper, 1000);

		function readURL(input) {
            if (input.files && input.files[0]) {
                var reader = new FileReader();
                
                reader.onload = function (e) {
                    $('#blah').attr('src', e.target.result)
                };
                reader.readAsDataURL(input.files[0]);
                setTimeout(initCropper, 1000);
            }
        };

	    function initCropper(){
	    	console.log("Came here")
	    	var image = document.getElementById('blah');
			var cropper = new Cropper(image, {
			  
			  crop: function(e) {

				document.getElementById("dataX").value = e.detail.x;
		  		document.getElementById("dataY").value = e.detail.y;
		  		document.getElementById("dataWidth").value = e.detail.width;
			  	document.getElementById("dataHeight").value = e.detail.height;
				document.getElementById("dataRotate").value = e.detail.rotate;
				document.getElementById("dataScaleX").value = e.detail.scaleX;
				document.getElementById("dataScaleY").value = e.detail.scaleY;
			
          
				desc=cropper.getData()
				console.log(JSON.stringify(desc))
				
				$("#img_cro").attr("value", JSON.stringify(desc));
				
		  
			    console.log(e.detail.x);
				
				var img=cropper.getCroppedCanvas({height:200}).toDataURL('image/jpeg')

				
				
			  	$("#uploader-preview").attr("src", img);



				
				
				
			    console.log(e.detail.y);
				
			  }
			});
			
			
			document.getElementById('zoom_button').addEventListener('click', function(){
			cropper.zoom(0.1)
			})
			document.getElementById('un_zoom_button').addEventListener('click', function(){
			cropper.zoom(-0.1)
			})
			document.getElementById('set_drag_mode').addEventListener('click', function(){
			cropper.setDragMode("move")
			
			})
			document.getElementById('set_crop_mode').addEventListener('click', function(){
			cropper.setDragMode("crop")
			
			})
			
			document.getElementById('move_left').addEventListener('click', function(){
			cropper.move(10,0)
			
			})
			document.getElementById('move_right').addEventListener('click', function(){
			cropper.move(10,0)
			
			})
			
			document.getElementById('move_up').addEventListener('click', function(){
			cropper.move(0,-10)
			
			})
			document.getElementById('move_down').addEventListener('click', function(){
			cropper.move(0,10)
			
			})
			
			
			document.getElementById('rotat_plus').addEventListener('click', function(){
			cropper.rotate(90)
			
			})
			
			
			document.getElementById('rotat_moin').addEventListener('click', function(){
			cropper.rotate(-90)
			
			})
			
			document.getElementById('scale_x').addEventListener('click', function(){
			console.log(document.getElementById("dataScaleX").value)
			if (document.getElementById("dataScaleX").value == -1) {
				cropper.scaleX(1)
			} else {
				cropper.scaleX(-1)
			}
			
			
			})
			document.getElementById('scale_y').addEventListener('click', function(){
			console.log(document.getElementById("dataScaleY").value)
			if (document.getElementById("dataScaleY").value  == -1) {
				cropper.scaleY(1)
			} else {
				cropper.scaleY(-1)
			}
			
			})
			
      
			
			
			
	    }

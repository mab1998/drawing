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
          
				
				var img=cropper.getCroppedCanvas({height:200}).toDataURL('image/jpeg')

			  	$("#uploader-preview").attr("src", img);
				
			  }
			  
			});
			
			
			
			
			var get_title=null
			document.getElementById("get_title").addEventListener("click", function(){
              //   console.log("Hello World");
                  document.getElementById("get_title").className = "btn btn-success";
				  
               //   console.log(cropper.getData())
                  if(get_title === null) {
                      get_title=cropper.getData()
					  document.getElementById("dra_title").value=JSON.stringify(get_title)
                  //    console.log(a)
                    }
                   else { 
                    cropper.setData(get_title)
                }})

			var drawing=null
		  document.getElementById("get_drawing").addEventListener("click", function(){
              //   console.log("Hello World");
                  document.getElementById("get_drawing").className = "btn btn-success";
				  
               //   console.log(cropper.getData())
                  if(drawing === null) {
                      drawing=cropper.getData()
					  document.getElementById("dra_number").value=JSON.stringify(drawing)
              //        console.log(desc)
                    }
                   else { 
                    cropper.setData(drawing)
                }})
                var get_project=null
        document.getElementById("get_project").addEventListener("click", function(){
              //   console.log("Hello World");
                  document.getElementById("get_project").className = "btn btn-success";
				  
               //   console.log(cropper.getData())
                  if(get_project === null) {
                    get_project=cropper.getData()
					document.getElementById("project_number").value=JSON.stringify(drawing)
					
                 //     console.log(Prise)
                    }
                   else { 
                    cropper.setData(get_project)
                }})
				
			 
				
				var revision=null
				document.getElementById("revision").addEventListener("click", function(){
              //   console.log("Hello World");
                  document.getElementById("revision").className = "btn btn-success";
               //   console.log(cropper.getData())
                  if(revision === null) {
                    revision=cropper.getData()
					document.getElementById("revsion").value=JSON.stringify(revision)
                 //     console.log(Prise)
                    }
                   else { 
                    cropper.setData(revision)
                }})	
				
				
				
				
				
				
		

          
            document.getElementById("reset").addEventListener("click", function(){
            //    console.log("Hello World");
            document.getElementById("get_title").className = "btn btn-info btn-arrow-right"
            document.getElementById("get_drawing").className = "btn btn-info btn-arrow-right"
            document.getElementById("get_project").className = "btn btn-info btn-arrow-right"
			document.getElementById("revision").className = "btn btn-info btn-arrow-right"
            document.getElementById("revsion").value=""
			document.getElementById("project_number").value=""
			document.getElementById("dra_number").value=""
			document.getElementById("dra_title").value=""
			 
             get_title=null
             drawing=null
             get_project=null
			 revision=null
            

            })
			
			
			

			
			
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
			document.getElementById('finish_btn').addEventListener('click', function(){
			var form = document.getElementById("finish");

			form.submit();
			
			})
			
			
      
			
			
			
	    }

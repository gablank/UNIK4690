# UNIK4690

## Index

	Main program: (ties all the pieces together, interactive interface)
      petanque_detection.py
    
    Run non-interactively and present detection statistics:
      batchrun.py
    
    Tool for manually label images with ball and playground positions: (used for detection evaluation)
      set_metadata.py
    
    Various utilities:
      utilities.py
      transformer.py
      image.py
    
    Playground detection methods:
      playground_detection/flood_fill.py
      playground_detection/red_balls.py
      playground_detection/manual_playground_detector.py (dummy to be able to work with ball detection only)
      playground_detection/maximize_density.py
        
    Ball detection methods:
      ball_detection/surf.py
      ball_detection/hough.py (early attempt)
      ball_detection/minimize_gradients.py (early attempt)
      
    
    Camera interfaces and image acquisition:
      camera.py
      lifecamstudiocamera.py
      networkcamera.py
      raspberry/picapture.py
      raspberry/raspberrycamera.py
      image_capture.py
    
    Simple tool to visualize a pipeline as various parameters are changed:
      pipeline_visualizer.py
      
    
    test.py
    
    report_helper.py
    genetic_algorithm.py
    demo/flood_fill_demonstration/flood_fill_demonstration.py
    demo/flood_fill_demonstration/transformer.py
    detect.py
    fitness.py
    histogram_explore.py
    resize.py


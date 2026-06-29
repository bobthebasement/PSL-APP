import anvil.server
import anvil.media
from anvil import *

class MainForm(MainFormTemplate):
    def __init__(self, **properties):
        # Set Form properties and Data Bindings
        self.init_components(**properties)
        
        # Initialize the form
        self.initialize_form()
    
    def initialize_form(self):
        """Initialize form state and load data"""
        # Set initial state
        self.analyze_button.enabled = False
        self.results_section.visible = False
        self.progress_bar.visible = False
        self.progress_bar.value = 0
        
        # Load leaderboard data
        self.load_leaderboard()
        
        # Load statistics
        self.load_statistics()
    
    def image_uploader_change(self, file, **event_args):
        """Handle image upload"""
        if file:
            self.analyze_button.enabled = True
            # Display the uploaded image
            self.uploaded_image.source = file
        else:
            self.analyze_button.enabled = False
            self.uploaded_image.source = None
    
    def analyze_button_click(self, **event_args):
        """Handle analyze button click"""
        # Disable the button and show progress
        self.analyze_button.enabled = False
        self.progress_bar.visible = True
        self.progress_bar.value = 0
        
        # Get the uploaded file
        uploaded_file = self.image_uploader.file
        
        if not uploaded_file:
            anvil.alert("Please upload a photo first!")
            self.analyze_button.enabled = True
            self.progress_bar.visible = False
            return
        
        # Read the image data
        try:
            image_data = uploaded_file.get_bytes()
            
            # Start the analysis in the background
            def update_progress(progress):
                self.progress_bar.value = progress
            
            def analysis_complete(result):
                self.progress_bar.visible = False
                
                if result.get('success', False):
                    self.display_results(result)
                else:
                    error_msg = result.get('error', 'An error occurred during analysis')
                    anvil.alert(f"Analysis failed: {error_msg}")
                    self.analyze_button.enabled = True
            
            # Call the server function
            anvil.server.call('rate_face', image_data, 
                            progress_handler=update_progress,
                            completion_handler=analysis_complete)
            
        except Exception as e:
            self.progress_bar.visible = False
            anvil.alert(f"Error processing image: {str(e)}")
            self.analyze_button.enabled = True
    
    def display_results(self, result):
        """Display the analysis results"""
        # Show the results section
        self.results_section.visible = True
        
        # Extract ratings (now 0-10)
        ratings = result.get('ratings', {})
        facial_geometry = ratings.get('facial_geometry', 0)
        feature_proportions = ratings.get('feature_proportions', 0)
        skin_quality = ratings.get('skin_quality', 0)
        dimorphism = ratings.get('dimorphism', 0)
        overall_score = result.get('overall_psl_score', 0)
        percentile = result.get('percentile', 0)
        improvements = result.get('improvements', [])
        
        # Update the UI with scores (out of 10)
        self.overall_score_value.text = f"{overall_score:.1f}/10"
        self.percentile_value.text = self.get_percentile_suffix(percentile)
        
        self.facial_geometry_score.text = f"{facial_geometry:.1f}/10"
        self.feature_proportions_score.text = f"{feature_proportions:.1f}/10"
        self.skin_quality_score.text = f"{skin_quality:.1f}/10"
        self.dimorphism_score.text = f"{dimorphism:.1f}/10"
        
        # Display improvements
        self.display_improvements(improvements)
        
        # Scroll to results
        self.main_container.scroll_to_component(self.results_section)
        
        # Reload leaderboard
        self.load_leaderboard()
        self.load_statistics()
        
        # Enable the new analysis button
        self.analyze_button.enabled = True
    
    def display_improvements(self, improvements):
        """Display the 3 improvements"""
        # Sort improvements by priority
        sorted_improvements = sorted(improvements, key=lambda x: x.get('priority', 0))
        
        # Update improvement labels
        for i, improvement in enumerate(sorted_improvements[:3]):
            category = improvement.get('category', 'General')
            improvement_text = improvement.get('improvement', 'No improvement suggestion')
            
            # Format the text with priority number and category
            formatted_text = f"{i+1}. [{category}] {improvement_text}"
            
            if i == 0:
                self.improvement_1_text.text = formatted_text
            elif i == 1:
                self.improvement_2_text.text = formatted_text
            elif i == 2:
                self.improvement_3_text.text = formatted_text
    
    def get_percentile_suffix(self, percentile):
        """Get the proper suffix for percentile"""
        if percentile == 1:
            return "1st"
        elif percentile == 2:
            return "2nd"
        elif percentile == 3:
            return "3rd"
        else:
            return f"{percentile:.0f}th"
    
    def new_analysis_button_click(self, **event_args):
        """Handle new analysis button click"""
        # Reset the form
        self.image_uploader.clear()
        self.uploaded_image.source = None
        self.analyze_button.enabled = False
        self.results_section.visible = False
        self.progress_bar.visible = False
        self.progress_bar.value = 0
        
        # Reset improvement texts
        self.improvement_1_text.text = '1. Loading...'
        self.improvement_2_text.text = '2. Loading...'
        self.improvement_3_text.text = '3. Loading...'
    
    def load_leaderboard(self):
        """Load leaderboard data"""
        try:
            leaderboard_data = anvil.server.call('get_leaderboard', 10)
            
            # Prepare data for the grid
            grid_data = []
            for i, entry in enumerate(leaderboard_data, 1):
                grid_data.append({
                    'Rank': i,
                    'Overall Score': f"{entry['overall_psl_score']:.1f}/10",
                    'Percentile': f"{entry['percentile']:.0f}th"
                })
            
            # Set the data
            self.leaderboard_grid.rows = grid_data
            
        except Exception as e:
            print(f"Error loading leaderboard: {e}")
            self.leaderboard_grid.rows = []
    
    def load_statistics(self):
        """Load statistics"""
        try:
            stats = anvil.server.call('get_statistics')
            # Could display stats in the footer or elsewhere
        except Exception as e:
            print(f"Error loading statistics: {e}")

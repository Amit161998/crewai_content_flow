#!/usr/bin/env python
"""
Guide Creator Flow - A comprehensive system for creating detailed guides on any topic.
This module implements a flow-based architecture using CrewAI to generate structured,
high-quality educational content with minimal user input.

The system follows a three-step process:
1. Gather user input about the topic and target audience
2. Create a structured outline using GPT-4
3. Generate detailed content for each section while maintaining context
"""

import json
import os
from typing import List, Dict
from pydantic import BaseModel, Field
from crewai import LLM
from crewai.flow.flow import Flow, listen, start
from guide_creator_flow.crews.content_crew.content_crew import ContentCrew

# Define Pydantic models for structured data validation and serialization


class Section(BaseModel):
    """
    Represents a single section of the guide with a title and description.
    Used for structuring the content outline.
    """
    title: str = Field(description="Title of the section")
    description: str = Field(
        description="Brief description of what the section should cover")


class GuideOutline(BaseModel):
    """
    Represents the complete structure of a guide including title, introduction,
    target audience, sections, and conclusion. This model is used both for
    generating the outline with GPT-4 and for storing the structure.
    """
    title: str = Field(description="Title of the guide")
    introduction: str = Field(description="Introduction to the topic")
    target_audience: str = Field(
        description="Description of the target audience")
    sections: List[Section] = Field(
        description="List of sections in the guide")
    conclusion: str = Field(description="Conclusion or summary of the guide")

# Define the state management class for the flow


class GuideCreatorState(BaseModel):
    """
    Maintains the state of the guide creation process throughout the flow.
    Stores the user's input, generated outline, and section contents.
    """
    topic: str = ""  # The main topic of the guide
    audience_level: str = ""  # Target audience level (beginner/intermediate/advanced)
    guide_outline: GuideOutline = None  # The structured outline of the guide
    sections_content: Dict[str, str] = {}  # Stores the content for each section


class GuideCreatorFlow(Flow[GuideCreatorState]):
    """
    Main flow class for creating comprehensive guides on any topic.
    Implements a step-by-step process from user input to final guide generation.
    Inherits from CrewAI's Flow class with GuideCreatorState as the state type.
    """

    @start()
    def get_user_input(self):
        """
        Initial step in the flow that collects user input.
        
        This method:
        1. Prompts the user for the guide topic
        2. Validates and collects the target audience level
        3. Updates the flow state with user inputs
        
        Returns:
            GuideCreatorState: Updated state with user inputs
        """
        print("\n=== Create Your Comprehensive Guide ===\n")

        # Get the main topic from the user
        self.state.topic = input(
            "What topic would you like to create a guide for? ")

        # Get and validate the audience level
        while True:
            audience = input(
                "Who is your target audience? (beginner/intermediate/advanced) ").lower()
            if audience in ["beginner", "intermediate", "advanced"]:
                self.state.audience_level = audience
                break
            print("Please enter 'beginner', 'intermediate', or 'advanced'")

        print(
            f"\nCreating a guide on {self.state.topic} for {self.state.audience_level} audience...\n")
        return self.state

    @listen(get_user_input)
    def create_guide_outline(self, state):
        """
        Second step in the flow that generates a structured outline using GPT-4.
        
        This method:
        1. Initializes the LLM with GPT-4
        2. Constructs a prompt for outline generation
        3. Processes the LLM response into a structured outline
        4. Saves the outline to a JSON file
        
        Args:
            state (GuideCreatorState): Current flow state with user inputs
            
        Returns:
            GuideOutline: Generated and structured outline for the guide
        """
        print("Creating guide outline...")

        # Initialize GPT-4 with structured output format
        llm = LLM(model="openai/gpt-4o-mini", response_format=GuideOutline)

        # Construct the system and user messages for the LLM
        messages = [
            {"role": "system",
                "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": f"""
            Create a detailed outline for a comprehensive guide on "{state.topic}" for {state.audience_level} level learners.

            The outline should include:
            1. A compelling title for the guide
            2. An introduction to the topic
            3. 4-6 main sections that cover the most important aspects of the topic
            4. A conclusion or summary

            For each section, provide a clear title and a brief description of what it should cover.
            """}
        ]

        # Generate the outline using GPT-4
        response = llm.call(messages=messages)

        # Convert JSON response to GuideOutline object
        outline_dict = json.loads(response)
        self.state.guide_outline = GuideOutline(**outline_dict)

        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)

        # Save the outline as JSON for reference
        with open("output/guide_outline.json", "w") as f:
            json.dump(outline_dict, f, indent=2)

        print(
            f"Guide outline created with {len(self.state.guide_outline.sections)} sections")
        return self.state.guide_outline

    @listen(create_guide_outline)
    def write_and_compile_guide(self, outline):
        """
        Final step in the flow that generates content for each section and compiles the guide.
        
        This method:
        1. Processes each section sequentially to maintain context
        2. Uses ContentCrew to generate detailed content for each section
        3. Maintains context by providing previously written sections
        4. Compiles all content into a final markdown document
        
        Args:
            outline (GuideOutline): The structured outline to generate content for
            
        Returns:
            str: Status message indicating completion
        """
        print("Writing guide sections and compiling...")
        completed_sections = []  # Track completed sections for context

        # Process sections one by one to maintain context flow
        for section in outline.sections:
            print(f"Processing section: {section.title}")

            # Build context from previous sections
            previous_sections_text = ""
            if completed_sections:
                previous_sections_text = "# Previously Written Sections\n\n"
                for title in completed_sections:
                    previous_sections_text += f"## {title}\n\n"
                    previous_sections_text += self.state.sections_content.get(
                        title, "") + "\n\n"
            else:
                previous_sections_text = "No previous sections written yet."

            # Run the content crew for this section
            result = ContentCrew().crew().kickoff(inputs={
                "section_title": section.title,
                "section_description": section.description,
                "audience_level": self.state.audience_level,
                "previous_sections": previous_sections_text,
                "draft_content": ""
            })

            # Store the content
            self.state.sections_content[section.title] = result.raw
            completed_sections.append(section.title)
            print(f"Section completed: {section.title}")

        # Compile the final guide
        guide_content = f"# {outline.title}\n\n"
        guide_content += f"## Introduction\n\n{outline.introduction}\n\n"

        # Add each section in order
        for section in outline.sections:
            section_content = self.state.sections_content.get(
                section.title, "")
            guide_content += f"\n\n{section_content}\n\n"

        # Add conclusion
        guide_content += f"## Conclusion\n\n{outline.conclusion}\n\n"

        # Save the guide
        with open("output/complete_guide.md", "w") as f:
            f.write(guide_content)

        print("\nComplete guide compiled and saved to output/complete_guide.md")
        return "Guide creation completed successfully"


def kickoff():
    """
    Entry point function to start the guide creation process.
    
    This function:
    1. Initializes and starts the GuideCreatorFlow
    2. Provides user feedback about completion
    3. Indicates where to find the generated guide
    """
    GuideCreatorFlow().kickoff()
    print("\n=== Flow Complete ===")
    print("Your comprehensive guide is ready in the output directory.")
    print("Open output/complete_guide.md to view it.")


def plot():
    """
    Utility function to visualize the flow structure.
    
    This function:
    1. Creates a new GuideCreatorFlow instance
    2. Generates a visual representation of the flow
    3. Saves the visualization as an HTML file
    """
    flow = GuideCreatorFlow()
    flow.plot("guide_creator_flow")
    print("Flow visualization saved to guide_creator_flow.html")


if __name__ == "__main__":
    kickoff()

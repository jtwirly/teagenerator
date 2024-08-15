import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
import json
import os
import re

# Get the OpenAI API key from the environment variable
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    st.error("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Function to parse the OpenAI response
def parse_openai_response(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        variables = []
        pattern = r"name\"?\s*:\s*\"?([^\"]+)\"?,\s*\"?min\"?\s*:\s*(\d+(?:\.\d+)?),\s*\"?max\"?\s*:\s*(\d+(?:\.\d+)?),\s*\"?default\"?\s*:\s*(\d+(?:\.\d+)?),\s*\"?step\"?\s*:\s*(\d+(?:\.\d+)?)"
        matches = re.findall(pattern, response_text)
        for match in matches:
            variables.append({
                "name": match[0],
                "min": float(match[1]),
                "max": float(match[2]),
                "default": float(match[3]),
                "step": max(0.1, float(match[4]))  # Ensure step is at least 0.1
            })
        return variables

# Function to generate business variables using OpenAI
def generate_business_variables(client, business_idea, location, assumptions, example_teas):
    prompt = f"""
    Based on the following business idea and context, generate 7-10 key variables for a Techno-Economic Analysis (TEA) tornado chart. Consider technical, economic, and environmental factors.

    Business Idea/Technology: {business_idea}
    Location: {location}
    Assumptions: {assumptions}
    Examples of other TEA analyses: {example_teas}

    For each variable, provide:
    1. Variable name
    2. Minimum value
    3. Maximum value
    4. Default value
    5. Step size for adjustment (must be at least 0.1)

    Consider variables that would be critical for a comprehensive TEA, including but not limited to:
    - Capital costs (CAPEX)
    - Operating costs (OPEX)
    - Revenue factors
    - Process efficiencies and yields
    - Environmental impact factors
    - Market size and growth rates
    - Regulatory costs or incentives
    - Raw material and feedstock costs
    - Technology readiness level (TRL) related factors

    Provide specific, realistic values based on current market data and industry standards. If exact data is not available, provide reasonable estimates.

    Respond in the following JSON format:
    [
        {{"name": "Variable 1", "min": 0, "max": 100, "default": 50, "step": 1}},
        {{"name": "Variable 2", "min": 0, "max": 100, "default": 50, "step": 0.1}},
        ...
    ]
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in Techno-Economic Analysis, capable of identifying key variables for various technologies and business ideas."},
            {"role": "user", "content": prompt}
        ]
    )
    
    variables = parse_openai_response(response.choices[0].message.content)
    
    # Ensure all variables have valid parameters and are of type float
    for var in variables:
        var['step'] = max(0.1, float(var['step']))
        var['min'] = float(min(var['min'], var['default']))
        var['max'] = float(max(var['max'], var['default']))
        var['default'] = float(var['default'])
        if var['min'] == var['max']:
            var['max'] += var['step']
    
    return variables

# Function to plot tornado chart
def plot_tornado_chart(variables, values):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(variables))
    sorted_indices = np.argsort(np.abs(np.array(values) - np.mean(values)))
    sorted_variables = [variables[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    
    ax.barh(y_pos, sorted_values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_variables)
    ax.invert_yaxis()
    ax.set_xlabel('Value')
    ax.set_title('Tornado Chart: Sensitivity Analysis')
    
    plt.tight_layout()
    return fig

# Function to generate comprehensive TEA report
def generate_tea_report(client, business_idea, location, assumptions, variables, values, example_teas, unit_of_interest):
    prompt = f"""
    You are an AI assistant specialized in conducting Techno-Economic Analyses (TEAs) for various technologies and processes. Your task is to provide a comprehensive TEA for the following business idea or technology:

    Business Idea/Technology: {business_idea}
    Location: {location}
    Assumptions: {assumptions}
    Unit of Interest: {unit_of_interest}
    Examples of other TEA analyses: {example_teas}

    Key Variables and Their Current Values:
    {', '.join([f"{var}: {val}" for var, val in zip(variables, values)])}

    Conduct a detailed TEA following these steps:

    1. Process Overview:
       - Summarize the key technical aspects of the proposed technology or business idea.
       - Identify the current Technology Readiness Level (TRL) and potential for advancement.
       - Create a simplified process flow diagram highlighting major steps and inputs/outputs.

    2. Market Assessment:
       - Consider the location provided.
       - Analyze the potential market size and growth trends.
       - Identify key competitors and substitutes.
       - Assess market drivers and barriers.

    3. Raw Materials and Feedstock:
       - List and describe all required raw materials and feedstocks.
       - Estimate costs for each, considering current market prices and potential future trends.
       - Assess the availability and sustainability of these inputs.

    4. Capital Expenditure (CAPEX) Estimation:
       - Estimate costs for major equipment items.
       - Include installation costs, engineering and design fees, and contingencies.
       - Provide a breakdown of total capital investment required.
       - Consider economies of scale and how CAPEX might change with facility size.

    5. Operating Expenditure (OPEX) Modeling:
       - Develop a comprehensive operating cost model, including:
         a) Raw materials and consumables
         b) Labor costs
         c) Utilities (energy, water, etc.)
         d) Maintenance and repairs
         e) Overhead and administrative costs
       - Provide a breakdown of costs per unit of product or service.

    6. Process Efficiency and Yield:
       - Estimate the overall process efficiency and product yield.
       - Identify key factors affecting yield and potential for improvement.

    7. Revenue Projections:
       - Estimate potential revenue based on market analysis and projected production capacity.
       - Consider different pricing scenarios and their impact on revenue.

    8. Profitability Analysis:
       - Calculate key financial metrics such as Net Present Value (NPV), Internal Rate of Return (IRR), and payback period.
       - Produce a tornado chart description.
       - Perform a discounted cash flow analysis.
       - Conduct a sensitivity analysis on key variables (e.g., feedstock cost, product price, yield).

    9. Environmental Impact Assessment:
       - Evaluate the potential environmental impacts of the technology or business.
       - Estimate energy requirements and greenhouse gas emissions.
       - Compare environmental performance to incumbent technologies or processes.

    10. Risk Assessment:
        - Identify key technical, market, and financial risks.
        - Suggest mitigation strategies for identified risks.

    11. Scalability and Technology Learning Curve:
        - Assess the scalability of the technology or business model.
        - Estimate potential cost reductions through learning curve effects or economies of scale.

    12. Regulatory Landscape:
        - Outline relevant regulations and standards applicable to the technology or business.
        - Discuss how the idea complies with or addresses these regulations.

    13. Co-Products and Waste Streams:
        - Identify and evaluate potential co-products or by-products.
        - Assess the marketability and value of these co-products.
        - Analyze waste streams and their disposal or potential valorization.

    14. Comparative Analysis:
        - Compare the estimated costs and projected performance to industry benchmarks or competing technologies.
        - Assess the competitive advantages and disadvantages of the proposed technology or business idea.

    15. Recommendations:
        - Provide 3-5 key recommendations for improving the technology's economic viability or reducing risks.
        - Suggest areas for further research or development.

    16. Executive Summary:
        - Summarize the key findings of your analysis.
        - Provide an overall assessment of the technology's or business idea's potential, including strengths, weaknesses, opportunities, and threats.
        - Provide a slide about TEA for an investor deck. Write the exact words for the slide so they can be copy-pasted out.

    17. Key Assumptions:
        - Make a list of the key assumptions used in this analysis.

    Please ensure your analysis is data-driven and technically sound. Be specific in your cost estimates and projections, clearly stating any assumptions made. If certain data is not available, indicate this and provide a range of estimates based on comparable technologies or industry standards.

    For the process flow diagram and cost models, provide detailed descriptions that could be used to create visual representations in external tools. Highlight any critical points or decision nodes in the process that significantly impact costs, efficiency, or feasibility.

    Your analysis should be thorough, objective, and actionable, providing valuable insights for decision-making regarding this technology or business idea. Consider both the technical feasibility and economic viability in your overall assessment. Use sensitivity analyses to identify the most impactful parameters on the overall economics.

    Format your response as a structured report with clear headings for each section, and provide specific numerical estimates where possible.
    
    Reference the structure and depth of analysis found in these example TEAs:
    - Direct Air Capture: https://pubs.acs.org/doi/10.1021/acs.est.0c00476
    - Renewable Hydrogen: https://pubs.rsc.org/en/content/articlehtml/2016/ee/c5ee02573g
    - https://www.sciencedirect.com/science/article/pii/S2542435121003032
    - https://www.sandia.gov/research/publications/details/techno-economic-analysis-best-practices-and-assessment-tools-2020-12-01/
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are an expert in Techno-Economic Analysis, capable of producing comprehensive and insightful reports with specific data and detailed breakdowns, following academic standards."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

# Streamlit app
st.title("TEA Generator")

# Get OpenAI API key
api_key = get_openai_api_key()

if api_key:
    client = OpenAI(api_key=api_key)

    # Main app
    business_idea = st.text_input("Enter your business idea/technology:")
    location = st.text_input("Enter the location:")
    unit_of_interest = st.selectbox("Select the unit of interest:", ["Product", "Process", "Operation"])
    assumptions = st.text_area("Enter key assumptions:")
    example_teas = st.text_area("Enter URLs of example TEA analyses:", value="""
    https://pubs.acs.org/doi/10.1021/acs.est.0c00476
    https://www.sciencedirect.com/science/article/pii/S2542435121003032
    https://pubs.rsc.org/en/content/articlehtml/2016/ee/c5ee02573g
    https://www.sandia.gov/research/publications/details/techno-economic-analysis-best-practices-and-assessment-tools-2020-12-01/
    """)

    if business_idea and location:
        if 'variables' not in st.session_state:
            with st.spinner("Generating variables for analysis..."):
                st.session_state.variables = generate_business_variables(client, business_idea, location, assumptions, example_teas)
        
        if st.session_state.variables:
            st.subheader("Adjust Key Variables for Sensitivity Analysis")
            values = []
            for var in st.session_state.variables:
                value = st.slider(
                    var['name'], 
                    min_value=float(var['min']),
                    max_value=float(var['max']),
                    value=float(var['default']),
                    step=float(var['step'])
                )
                values.append(value)
            
            if values:
                st.subheader("Tornado Chart: Sensitivity Analysis")
                fig = plot_tornado_chart([var['name'] for var in st.session_state.variables], values)
                st.pyplot(fig)
                
                if st.button("Generate Comprehensive TEA Report"):
                    with st.spinner("Generating TEA report... This may take a few minutes."):
                        tea_report = generate_tea_report(client, business_idea, location, assumptions, 
                                                        [var['name'] for var in st.session_state.variables], 
                                                        values, example_teas, unit_of_interest)
                        st.markdown("## Techno-Economic Analysis Report")
                        st.markdown(tea_report)
        else:
            st.error("Failed to generate variables. Please try again or refine your business idea.")
    else:
        st.info("Please enter your business idea/technology and location to begin the analysis.")
else:
    st.warning("Please provide an OpenAI API key to use this app.")
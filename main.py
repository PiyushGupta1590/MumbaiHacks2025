import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from pathlib import Path
from dotenv import load_dotenv
import time
import traceback

# Import your existing crew and email service
from crew import Projectneon
from email_service import send_reports_via_email

# --- Configuration & Setup ---
# Load environment variables from a .env file in the same directory
load_dotenv()

# Verify that the necessary API key is loaded
if not os.getenv("NVIDIA_NIM_API_KEY"):
    print("⚠️ WARNING: NVIDIA_NIM_API_KEY not found. The AI analysis will fail.")
    print("   Please create a .env file in the root of your project with your key.")


# --- Helper Function: Process Uploaded Excel File ---
def process_excel(file):
    """
    Reads and validates the uploaded Excel file, then calculates the running balance.
    """
    if file is None:
        return "Please upload a valid Excel file.", None

    try:
        df = pd.read_excel(file.name)
        required_cols = ['Date', 'Party Name', 'Cash Inflow', 'Cash Outflow', 'Payment Status']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            error_message = f"❌ Error: The uploaded file is missing required columns: **{', '.join(missing_cols)}**"
            return error_message, None

        # Standardize data types and calculate running balance
        df['Date'] = pd.to_datetime(df['Date'])
        if 'Running Balance' not in df.columns:
            df = df.sort_values('Date').reset_index(drop=True)
            df['Running Balance'] = (
                df['Cash Inflow'].fillna(0) - df['Cash Outflow'].fillna(0).abs()
            ).cumsum()

        # Create a summary message for the UI
        summary = (
            f"✅ **File processed successfully!**\n\n"
            f"**Total Transactions:** {len(df)}\n"
            f"**Total Inflow:** ${df['Cash Inflow'].sum():,.2f}\n"
            f"**Total Outflow:** ${df['Cash Outflow'].sum():,.2f}\n"
            f"**Current Balance:** ${df['Running Balance'].iloc[-1]:,.2f}"
        )
        return summary, df

    except Exception as e:
        return f"❌ An error occurred while processing the file: {e}", None


# --- Helper Function: Run the CrewAI Analysis ---
def run_ai_analysis(df, owner_email, send_email):
    """
    Initializes and runs the Projectneon crew, then returns the paths to the generated reports.
    """
    if df is None:
        return "❌ Please upload and process your data first.", None, None, None

    if not os.getenv("NVIDIA_NIM_API_KEY"):
        return "❌ **CRITICAL ERROR:** NVIDIA_NIM_API_KEY not found!", None, None, None

    try:
        # 1. Initialize the crew
        projectneon = Projectneon()

        # 2. Generate the initial text-based cash flow report.
        # This is a crucial step as it populates the class variable that tasks will use.
        projectneon.generate_cash_flow_analysis(df)
        
        # 3. Define the inputs for the crew kickoff
        inputs = {
            "topic": "Comprehensive Financial Health Assessment & Cash Flow Analysis",
            "dataset": "Uploaded Excel data",
        }
        
        # 4. Kick off the crew to generate the reports
        result = projectneon.crew().kickoff(inputs=inputs)
        print("Crew execution result:", result) # For debugging

        # 5. Define the paths to the generated report files
        report_dir = Path(__file__).parent
        reports = {
            "cash_flow": report_dir / "cash_flow_analysis.md",
            "risk": report_dir / "risk_assessment.md",
            # CORRECTED: This now matches the output file from your communications_task
            "executive_summary": report_dir / "financial_health_report.md" 
        }

        # Ensure all files were actually created before proceeding
        for key, path in reports.items():
            if not path.exists():
                error_msg = f"❌ Error: Expected report file '{path.name}' was not created by the crew."
                return error_msg, None, None, None

        # 6. Send email if requested
        if send_email:
            print(f"Sending email to {owner_email}...")
            send_reports_via_email(
                report_files=[str(p) for p in reports.values()],
                owner_email=owner_email
            )
            print("Email function executed.")

        # 7. Return success message and file paths for Gradio to display
        return (
            "✅ **Analysis complete!** Reports have been generated below.",
            str(reports["cash_flow"]),
            str(reports["risk"]),
            str(reports["executive_summary"])
        )
    except Exception as e:
        # Provide detailed error feedback to the user
        error_details = traceback.format_exc()
        return f"❌ An unexpected error occurred during AI analysis: {e}\n\nTraceback:\n{error_details}", None, None, None


# --- Helper Function: Generate Dashboard Visuals ---
def generate_dashboard(df):
    """
    Creates Plotly charts from the processed DataFrame.
    """
    if df is None:
        # Return empty figures if no data is present
        empty_fig = go.Figure().update_layout(title_text="No Data Available")
        return empty_fig, empty_fig, empty_fig, empty_fig

    # Chart 1: Cash Flow Trend (Running Balance over time)
    fig1 = px.line(df, x='Date', y='Running Balance', title="📈 Cash Flow Trend (Running Balance)", markers=True)
    fig1.update_traces(line=dict(color='#667eea', width=3), fill='tozeroy')
    fig1.update_layout(xaxis_title="Date", yaxis_title="Balance ($)")

    # Chart 2: Monthly Inflow vs Outflow
    df_monthly = df.copy()
    df_monthly['Month'] = df_monthly['Date'].dt.to_period('M').astype(str)
    monthly_agg = df_monthly.groupby('Month').agg({'Cash Inflow': 'sum', 'Cash Outflow': 'sum'}).reset_index()
    fig2 = go.Figure(data=[
        go.Bar(name='Inflow', x=monthly_agg['Month'], y=monthly_agg['Cash Inflow'], marker_color='#28a745'),
        go.Bar(name='Outflow', x=monthly_agg['Month'], y=monthly_agg['Cash Outflow'], marker_color='#dc3545')
    ])
    fig2.update_layout(barmode='group', title="💰 Monthly Inflow vs Outflow")

    # Chart 3: Payment Status Distribution
    fig3 = px.pie(df, names='Payment Status', title="📊 Payment Status Distribution",
                  color='Payment Status',
                  color_discrete_map={'Paid': '#28a745', 'Pending': '#ffc107', 'Overdue': '#dc3545'})

    # Chart 4: Top 5 Customers by Inflow
    inflows_df = df[df['Cash Inflow'] > 0]
    top_customers = inflows_df.groupby('Party Name')['Cash Inflow'].sum().nlargest(5).sort_values()
    fig4 = px.bar(top_customers, x=top_customers.values, y=top_customers.index, orientation='h',
                  title="👥 Top 5 Customers by Revenue", labels={'x': 'Total Inflow ($)', 'y': 'Customer'})
    fig4.update_traces(marker_color='#667eea')

    return fig1, fig2, fig3, fig4


# --- Build Gradio User Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="💰 Financial Health Analyzer") as demo:
    gr.Markdown("# 💰 Financial Health Analyzer")
    gr.Markdown("An AI-powered dashboard for cash flow and risk analysis using **NVIDIA NIM** and **CrewAI**.")

    # Hidden state to hold the processed DataFrame
    df_state = gr.State()

    with gr.Tabs() as tabs:
        # Tab 1: Data Upload
        with gr.TabItem("1. Upload Data", id=0):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload Financial Data (.xlsx)", file_types=['.xlsx'])
                    upload_btn = gr.Button("📂 Process File", variant="primary")
                with gr.Column(scale=2):
                    upload_output = gr.Markdown(value="*Please upload your Excel file to begin.*")
        
        # Tab 2: Interactive Dashboard
        with gr.TabItem("2. Dashboard", id=1):
            with gr.Row():
                plot1 = gr.Plot(label="Cash Flow Trend")
                plot2 = gr.Plot(label="Inflow vs Outflow")
            with gr.Row():
                plot3 = gr.Plot(label="Payment Status")
                plot4 = gr.Plot(label="Top Customers")

        # Tab 3: AI-Powered Analysis & Reports
        with gr.TabItem("3. AI Analysis & Reports", id=2):
            with gr.Row():
                with gr.Column(scale=1):
                    owner_email = gr.Textbox(label="Business Owner Email (for reports)", value="owner@example.com")
                    send_email = gr.Checkbox(label="Send Final Reports via Email", value=False)
                    analyze_btn = gr.Button("🚀 Start AI Analysis", variant="primary")
                with gr.Column(scale=2):
                    analysis_output = gr.Markdown("*AI analysis results will appear here.*")
            
            gr.Markdown("---")
            gr.Markdown("### Generated Reports")
            with gr.Row():
                # CORRECTED: Changed the label to reflect the actual report content
                report_exec_summary = gr.File(label="Executive Summary Report")
                report_risk = gr.File(label="Risk Assessment Report")
                report_cashflow = gr.File(label="Detailed Cash Flow Analysis")

    # --- Event Handlers ---

    # Action for the "Process File" button
    def process_and_generate_dashboard(file):
        summary, df = process_excel(file)
        fig1, fig2, fig3, fig4 = generate_dashboard(df)
        # Switch to the dashboard tab after processing
        return summary, df, fig1, fig2, fig3, fig4, gr.Tabs(selected=1)

    upload_btn.click(
        fn=process_and_generate_dashboard,
        inputs=file_input,
        outputs=[upload_output, df_state, plot1, plot2, plot3, plot4, tabs]
    )

    # Action for the "Start AI Analysis" button
    def start_analysis_and_switch_tab(df, email, should_send):
        # This function will call the main analysis logic
        # and then return the updated UI components and the tab selection
        msg, f1, f2, f3 = run_ai_analysis(df, email, should_send)
        return msg, f1, f2, f3

    analyze_btn.click(
        fn=start_analysis_and_switch_tab,
        inputs=[df_state, owner_email, send_email],
        outputs=[analysis_output, report_cashflow, report_risk, report_exec_summary]
    )

    # Footer
    gr.Markdown(
        "<div style='text-align:center; color:gray; margin-top: 20px;'>"
        "<b>SME Monitor Dashboard</b> | Powered by CrewAI & NVIDIA NIM<br>"
        "</div>"
    )

if __name__ == "__main__":
    demo.launch(debug=True)

#!/usr/bin/env python
import sys
import warnings
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import os
import pandas as pd

from crew import Projectneon

# Define the path to the Excel file
excel_file_path = Path(r'D:\sme_montior\neon\knowledge\financial_data.xlsx')

env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the Financial Analysis Crew with three agents:
    1. Cash Flow Analyst - Analyzes transactions and calculates cash metrics
    2. Risk Analyst - Assesses financial risks and identifies urgent actions
    3. Communications Manager - Drafts communications and compiles executive report
    """
    if not os.getenv("NVIDIA_NIM_API_KEY"):
        print("❌ Error: NVIDIA_NIM_API_KEY not found")
        return
    
    print(f"✓ Using model: {os.getenv('MODEL', 'nvidia_nim/meta/llama-3.1-405b-instruct')}")
    
    # Read the Excel file
    try:
        print(f"📂 Reading {excel_file_path}...")
        df = pd.read_excel(excel_file_path)
        
        # Preliminary analysis for context
        total_inflow = df[df['Cash Inflow'] > 0]['Cash Inflow'].sum()
        total_outflow = df[df['Cash Outflow'] < 0]['Cash Outflow'].sum()
        net_cash_flow = total_inflow + total_outflow
        
        overdue_count = len(df[df['Payment Status'] == 'Overdue'])
        overdue_amount = df[df['Payment Status'] == 'Overdue']['Cash Inflow'].sum()
        pending_count = len(df[df['Payment Status'] == 'Pending'])
        paid_count = len(df[df['Payment Status'] == 'Paid'])
        
        client_transactions = len(df[df['Party Type'] == 'Client'])
        supplier_transactions = len(df[df['Party Type'] == 'Supplier'])
        
        high_priority_pending = len(df[(df['Payment Status'] == 'Pending') & (df['Priority'] == 'High')])
        
        # Enhanced data summary for AI agents
        data_summary = f"""
📊 FINANCIAL TRANSACTION DATASET - COMPLETE ANALYSIS

Dataset Metadata:
- Total Transactions: {len(df)}
- Date Range: {df['Date'].min()} to {df['Date'].max()}
- Columns: {', '.join(df.columns.tolist())}

💰 CASH FLOW SUMMARY:
- Total Cash Inflow: ${total_inflow:,.2f}
- Total Cash Outflow: ${total_outflow:,.2f}
- Net Cash Flow: ${net_cash_flow:,.2f}
- Average Transaction Size: ${df['Cash Inflow'].replace(0, pd.NA).mean():,.2f}

📋 TRANSACTION BREAKDOWN:
- Client Transactions (Sales): {client_transactions}
- Supplier Transactions (Purchases): {supplier_transactions}

💳 PAYMENT STATUS ANALYSIS:
- ✅ Paid: {paid_count} transactions
- ⏳ Pending: {pending_count} transactions
- ⚠️ Overdue: {overdue_count} transactions (Total: ${overdue_amount:,.2f})
- 🔴 High Priority Pending: {high_priority_pending} transactions

📂 TRANSACTION CATEGORIES:
{df['Category'].value_counts().to_string()}

🎯 PRIORITY DISTRIBUTION:
{df['Priority'].value_counts().to_string()}

📋 COMPLETE DATASET (All Rows):
{df.to_string()}

📊 DATA TYPES:
{df.dtypes.to_string()}

⚠️ MISSING VALUES:
{df.isnull().sum().to_string()}

📈 STATISTICAL SUMMARY:
{df.describe().to_string()}

🔍 TOP 10 LARGEST TRANSACTIONS:
{df.nlargest(10, 'Cash Inflow')[['Party Name', 'Cash Inflow', 'Payment Status', 'Priority']].to_string()}

⚠️ ALL OVERDUE TRANSACTIONS:
{df[df['Payment Status'] == 'Overdue'][['Party Name', 'Cash Inflow', 'Date', 'Notes']].to_string()}
"""
        
        print("✓ Excel file loaded successfully!")
        print(f"\n📊 Quick Stats:")
        print(f"  - Total Transactions: {len(df)}")
        print(f"  - Net Cash Flow: ${net_cash_flow:,.2f}")
        print(f"  - Overdue Transactions: {overdue_count} (${overdue_amount:,.2f})")
        print(f"  - High Priority Pending: {high_priority_pending}")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found at {excel_file_path}")
        print(f"Please ensure the file exists at: {excel_file_path.absolute()}")
        return
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Prepare inputs for the crew
    inputs = {
        "topic": "Comprehensive Financial Health Assessment & Cash Flow Analysis",
        "dataset": str(excel_file_path),
        "financial_data": data_summary,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_transactions": len(df),
        "net_cash_flow": f"${net_cash_flow:,.2f}",
        "overdue_count": overdue_count,
        "overdue_amount": f"${overdue_amount:,.2f}"
    }

    print("\n🚀 Starting Financial Analysis Crew...\n")
    print("👥 Crew Members:")
    print("  1. Cash Flow Analyst - Analyzing transactions and cash metrics")
    print("  2. Risk Analyst - Assessing financial risks and priorities")
    print("  3. Communications Manager - Drafting reports and communications\n")

    try:
        crew = Projectneon().crew()
        result = crew.kickoff(inputs=inputs)

        print("\n" + "="*80)
        print("✅ CREW EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n📊 Final Result:")
        print(result)
        print("\n📁 Output Files Generated:")
        print("  - financial_health_report.md (Executive Report)")
        print("  - overdue_report.xlsx (Overdue Invoice List)")
        print("  - overdue_communications.txt (Payment Reminders)")

    except Exception as e:
        print(f"\n⚠️ An error occurred while running the crew:\n{e}")
        import traceback
        traceback.print_exc()


def train():
    """
    Train the crew for 5 iterations.
    """
    inputs = {
        "topic": "Financial Transaction Analysis Training"
    }
    try:
        Projectneon().crew().train(n_iterations=5, inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Projectneon().crew().replay(task_id="cash_flow_analysis_task")
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "Financial Transaction Analysis Test"
    }
    try:
        Projectneon().crew().test(n_iterations=1, openai_model_name="gpt-4", inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


if __name__ == "__main__":
    run()
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from pathlib import Path
import yaml
import os
import pandas as pd
from datetime import datetime
from io import StringIO

@CrewBase
class Projectneon():
    """Projectneon crew for financial transaction analysis"""
    
    # Store the cash flow report as instance variable
    cash_flow_report_data = None

    def __init__(self):
        """Load YAML configs for agents and tasks"""
        config_dir = Path(__file__).parent / "config"

        # Load Agents Config
        with open(config_dir / "agents.yaml", "r", encoding="utf-8") as f:
            agents_data = yaml.safe_load(f)
            self.agents_config = agents_data.get("agents", {})

        # Load Tasks Config
        with open(config_dir / "tasks.yaml", "r", encoding="utf-8") as f:
            tasks_data = yaml.safe_load(f)
            self.tasks_config = tasks_data.get("tasks", {})
        
        # Configure NVIDIA NIM LLM
        self.llm = LLM(
            model="nvidia_nim/meta/llama-3.1-405b-instruct",
            api_key=os.getenv("NVIDIA_NIM_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1",
            timeout=300
        )

    def generate_cash_flow_analysis(self, df):
        """Generate comprehensive cash flow analysis report"""
        output = StringIO()
        
        def write(text):
            output.write(text + "\n")
        
        current_balance = df['Running Balance'].iloc[-1] if len(df) > 0 else 0
        latest_date = df['Date'].iloc[-1] if len(df) > 0 else datetime.now()
        
        write("\n" + "="*80)
        write("CASH POSITION SNAPSHOT")
        write("="*80)
        write(f"Current Cash Balance: ${current_balance:,.2f}")
        write(f"As of Date: {latest_date.strftime('%Y-%m-%d')}")
        write(f"Total Transactions Processed: {len(df)}")
        
        # Inflow Analysis
        inflows = df[df['Cash Inflow'] > 0].copy() if 'Cash Inflow' in df.columns else pd.DataFrame()
        total_inflow = inflows['Cash Inflow'].sum() if len(inflows) > 0 else 0
        paid_inflow = inflows[inflows['Payment Status'] == 'Paid']['Cash Inflow'].sum() if len(inflows) > 0 else 0
        pending_inflow = inflows[inflows['Payment Status'] == 'Pending']['Cash Inflow'].sum() if len(inflows) > 0 else 0
        overdue_inflow = inflows[inflows['Payment Status'] == 'Overdue']['Cash Inflow'].sum() if len(inflows) > 0 else 0
        
        write("\n" + "="*80)
        write("INFLOW ANALYSIS (Revenue)")
        write("="*80)
        write(f"Total Inflows: ${total_inflow:,.2f}")
        if total_inflow > 0:
            write(f"  - Paid: ${paid_inflow:,.2f} ({paid_inflow/total_inflow*100:.1f}%)")
            write(f"  - Pending: ${pending_inflow:,.2f} ({pending_inflow/total_inflow*100:.1f}%)")
            write(f"  - Overdue: ${overdue_inflow:,.2f} ({overdue_inflow/total_inflow*100:.1f}%)")
        
        # Top 10 Customers
        if len(inflows) > 0:
            top_customers = inflows.groupby('Party Name')['Cash Inflow'].sum().sort_values(ascending=False).head(10)
            write(f"\nTop 10 Customers by Revenue:")
            write(f"{'Rank':<6} {'Customer Name':<40} {'Amount':<15} {'% of Total':<10}")
            write("-" * 71)
            for rank, (customer, amount) in enumerate(top_customers.items(), 1):
                pct = (amount / total_inflow) * 100 if total_inflow > 0 else 0
                write(f"{rank:<6} {customer:<40} ${amount:>12,.2f}  {pct:>7.1f}%")
            
            # Payment Status by Customer
            write(f"\nTop 10 Customers - Payment Status Breakdown:")
            write(f"{'Customer Name':<40} {'Paid':<15} {'Pending':<15} {'Overdue':<15}")
            write("-" * 85)
            for customer in top_customers.index:
                customer_data = inflows[inflows['Party Name'] == customer]
                paid = customer_data[customer_data['Payment Status'] == 'Paid']['Cash Inflow'].sum()
                pending = customer_data[customer_data['Payment Status'] == 'Pending']['Cash Inflow'].sum()
                overdue = customer_data[customer_data['Payment Status'] == 'Overdue']['Cash Inflow'].sum()
                write(f"{customer:<40} ${paid:>12,.2f}  ${pending:>12,.2f}  ${overdue:>12,.2f}")
        
        # Outflow Analysis
        outflows = df[df['Cash Outflow'] > 0].copy() if 'Cash Outflow' in df.columns else pd.DataFrame()
        total_outflow = outflows['Cash Outflow'].sum() if len(outflows) > 0 else 0
        paid_outflow = outflows[outflows['Payment Status'] == 'Paid']['Cash Outflow'].sum() if len(outflows) > 0 else 0
        pending_outflow = outflows[outflows['Payment Status'] == 'Pending']['Cash Outflow'].sum() if len(outflows) > 0 else 0
        overdue_outflow = outflows[outflows['Payment Status'] == 'Overdue']['Cash Outflow'].sum() if len(outflows) > 0 else 0
        
        write("\n" + "="*80)
        write("OUTFLOW ANALYSIS (Expenses)")
        write("="*80)
        write(f"Total Outflows: ${total_outflow:,.2f}")
        if total_outflow > 0:
            write(f"  - Paid: ${paid_outflow:,.2f} ({paid_outflow/total_outflow*100:.1f}%)")
            write(f"  - Pending: ${pending_outflow:,.2f} ({pending_outflow/total_outflow*100:.1f}%)")
            write(f"  - Overdue: ${overdue_outflow:,.2f} ({overdue_outflow/total_outflow*100:.1f}%)")
        
        # Top Vendors
        if len(outflows) > 0:
            top_vendors = outflows.groupby('Party Name')['Cash Outflow'].sum().sort_values(ascending=False).head(10)
            write(f"\nTop 10 Vendors by Payables:")
            write(f"{'Rank':<6} {'Vendor Name':<40} {'Amount':<15} {'% of Total':<10}")
            write("-" * 71)
            for rank, (vendor, amount) in enumerate(top_vendors.items(), 1):
                pct = (amount / total_outflow) * 100 if total_outflow > 0 else 0
                write(f"{rank:<6} {vendor:<40} ${amount:>12,.2f}  {pct:>7.1f}%")
        
        # Overdue Receivables
        overdue = df[(df['Cash Inflow'] > 0) & (df['Payment Status'] == 'Overdue')].copy() if len(df) > 0 else pd.DataFrame()
        total_overdue = overdue['Cash Inflow'].sum() if len(overdue) > 0 else 0
        
        write("\n" + "="*80)
        write("OVERDUE RECEIVABLES ANALYSIS")
        write("="*80)
        write(f"Total Overdue Amount: ${total_overdue:,.2f}")
        write(f"Number of Overdue Invoices: {len(overdue)}")
        if len(overdue) > 0:
            write(f"Average Overdue Amount: ${overdue['Cash Inflow'].mean():,.2f}")
            write(f"\nOverdue Invoices Detail:")
            write(f"{'Customer Name':<40} {'Amount':<15} {'Date':<15} {'Days Overdue':<12}")
            write("-" * 82)
            for idx, row in overdue.iterrows():
                days_overdue = (pd.Timestamp.now() - row['Date']).days
                write(f"{row['Party Name']:<40} ${row['Cash Inflow']:>12,.2f}  {row['Date'].strftime('%Y-%m-%d'):<15} {days_overdue:>10} days")
        
        # Pending Payables
        pending = df[(df['Cash Outflow'] > 0) & (df['Payment Status'] == 'Pending')].copy() if len(df) > 0 else pd.DataFrame()
        total_pending = pending['Cash Outflow'].sum() if len(pending) > 0 else 0
        
        write("\n" + "="*80)
        write("PENDING PAYABLES ANALYSIS")
        write("="*80)
        write(f"Total Pending Payables: ${total_pending:,.2f}")
        write(f"Number of Pending Obligations: {len(pending)}")
        if len(pending) > 0:
            write(f"Average Pending Amount: ${pending['Cash Outflow'].mean():,.2f}")
            write(f"\nPending Payables Detail:")
            write(f"{'Vendor Name':<40} {'Amount':<15} {'Date':<15} {'Days Outstanding':<15}")
            write("-" * 85)
            for idx, row in pending.iterrows():
                days_outstanding = (pd.Timestamp.now() - row['Date']).days
                write(f"{row['Party Name']:<40} ${row['Cash Outflow']:>12,.2f}  {row['Date'].strftime('%Y-%m-%d'):<15} {days_outstanding:>13} days")
        
        # Key Metrics
        if len(df) > 1:
            date_range = max((df['Date'].max() - df['Date'].min()).days, 1)
            daily_burn = (total_outflow - total_inflow) / date_range
            
            if daily_burn > 0:
                cash_runway_days = current_balance / daily_burn if daily_burn > 0 else float('inf')
            else:
                cash_runway_days = float('inf')
        else:
            daily_burn = 0
            cash_runway_days = float('inf')
        
        write("\n" + "="*80)
        write("KEY FINANCIAL METRICS")
        write("="*80)
        write(f"Total Cash Inflows: ${total_inflow:,.2f}")
        write(f"Total Cash Outflows: ${total_outflow:,.2f}")
        write(f"Net Cash Flow: ${total_inflow - total_outflow:,.2f}")
        write(f"Current Cash Balance: ${current_balance:,.2f}")
        write(f"Daily Burn Rate: ${daily_burn:,.2f}")
        if cash_runway_days != float('inf'):
            write(f"Cash Runway: {cash_runway_days:.1f} days")
        else:
            write(f"Cash Runway: Positive cash generation (indefinite)")
        write(f"Working Capital Ratio: {(total_inflow / total_outflow):.2f}x" if total_outflow > 0 else "Working Capital Ratio: N/A")
        
        report = output.getvalue()
        # Store in instance variable for tasks to access
        Projectneon.cash_flow_report_data = report
        return report

    @agent
    def cash_analyst(self) -> Agent:
        config = self.agents_config.get("cash_analyst", {})
        return Agent(
            role=config.get("role", "Cash Flow Analyst"),
            goal=config.get("goal", "Analyze financial transactions and cash flow"),
            backstory=config.get("backstory", "Expert in cash flow analysis"),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    @agent
    def risk_analyst(self) -> Agent:
        config = self.agents_config.get("risk_analyst", {})
        return Agent(
            role=config.get("role", "Credit Risk Analyst"),
            goal=config.get("goal", "Assess financial health and identify risks"),
            backstory=config.get("backstory", "Expert in financial risk assessment"),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    @agent
    def communications_manager(self) -> Agent:
        config = self.agents_config.get("communications_manager", {})
        return Agent(
            role=config.get("role", "Communications & Action Manager"),
            goal=config.get("goal", "Draft communications and compile reports"),
            backstory=config.get("backstory", "Expert in financial communications"),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    @task
    def cash_flow_analysis_task(self) -> Task:
        config = self.tasks_config.get("cash_flow_analysis_task", {})
        # Embed the actual data in the description
        description = f"""{config.get("description", "Analyze the cash flow data")}

HERE IS THE ACTUAL FINANCIAL DATA TO ANALYZE:

{Projectneon.cash_flow_report_data or "No data available"}

IMPORTANT: Analyze THIS ACTUAL DATA above. Extract specific numbers, customer names, amounts, and payment statuses. Do NOT generate generic answers."""
        
        return Task(
            description=description,
            expected_output=config.get("expected_output", "Detailed cash flow analysis with specific customer names, amounts, and metrics from the data provided"),
            agent=self.cash_analyst(),
            output_file='cash_flow_analysis.md'
        )

    @task
    def risk_assessment_task(self) -> Task:
        config = self.tasks_config.get("risk_assessment_task", {})
        description = f"""{config.get("description", "Assess financial risks")}

REFERENCE DATA:
{Projectneon.cash_flow_report_data or "No data available"}

IMPORTANT: Use the specific customer names, overdue amounts, and pending payables from the data above. List actual customers with actual amounts they owe. Do NOT generate generic examples."""
        
        return Task(
            description=description,
            expected_output=config.get("expected_output", "Risk assessment with specific customer names, exact amounts owed/pending, and prioritized actions"),
            agent=self.risk_analyst(),
            output_file='risk_assessment.md',
            context=[self.cash_flow_analysis_task()]
        )

    @task
    def communications_task(self) -> Task:
        config = self.tasks_config.get("communications_task", {})
        description = f"""{config.get("description", "Create executive report")}

FINANCIAL DATA:
{Projectneon.cash_flow_report_data or "No data available"}

IMPORTANT: 
- Use REAL customer names from the data, not "Customer XYZ" or "Customer ABC"
- Include ACTUAL amounts from the financial data
- Reference specific overdue customers and pending payments
- Provide concrete action items based on real data"""
        
        return Task(
            description=description,
            expected_output=config.get("expected_output", "Executive report with real customer names, specific amounts, and actionable recommendations based on actual financial data"),
            agent=self.communications_manager(),
            output_file='financial_health_report.md',
            context=[self.cash_flow_analysis_task(), self.risk_assessment_task()]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Projectneon crew with all three agents"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=False
        )
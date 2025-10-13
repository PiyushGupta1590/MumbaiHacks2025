from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from pathlib import Path
import yaml
import os

@CrewBase
class Projectneon():
    """Projectneon crew for financial transaction analysis"""

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
            base_url="https://integrate.api.nvidia.com/v1"
        )

    @agent
    def cash_analyst(self) -> Agent:
        config = self.agents_config.get("cash_analyst", {})
        return Agent(
            role=config.get("role", "Cash Flow Analyst"),
            goal=config.get("goal", "Analyze financial transactions and cash flow"),
            backstory=config.get("backstory", "Expert in cash flow analysis"),
            llm=self.llm,
            verbose=True
        )

    @agent
    def risk_analyst(self) -> Agent:
        config = self.agents_config.get("risk_analyst", {})
        return Agent(
            role=config.get("role", "Credit Risk Analyst"),
            goal=config.get("goal", "Assess financial health and identify risks"),
            backstory=config.get("backstory", "Expert in financial risk assessment"),
            llm=self.llm,
            verbose=True
        )

    @agent
    def communications_manager(self) -> Agent:
        config = self.agents_config.get("communications_manager", {})
        return Agent(
            role=config.get("role", "Communications & Action Manager"),
            goal=config.get("goal", "Draft communications and compile reports"),
            backstory=config.get("backstory", "Expert in financial communications"),
            llm=self.llm,
            verbose=True
        )

    @task
    def cash_flow_analysis_task(self) -> Task:
        config = self.tasks_config.get("cash_flow_analysis_task", {})
        return Task(
            description=config.get("description", "Analyze financial transactions and cash flow"),
            expected_output=config.get("expected_output", "Comprehensive cash flow analysis"),
            agent=self.cash_analyst()
        )

    @task
    def risk_assessment_task(self) -> Task:
        config = self.tasks_config.get("risk_assessment_task", {})
        return Task(
            description=config.get("description", "Assess financial risks and identify urgent actions"),
            expected_output=config.get("expected_output", "Detailed risk assessment report"),
            agent=self.risk_analyst()
        )

    @task
    def communications_task(self) -> Task:
        config = self.tasks_config.get("communications_task", {})
        return Task(
            description=config.get("description", "Draft communications and compile executive report"),
            expected_output=config.get("expected_output", "Executive report and communications"),
            agent=self.communications_manager(),
            output_file='financial_health_report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Projectneon crew with all three agents"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
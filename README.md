📚 ATOMIC CHRONICLE v4.0: A World You Want — Or a World of Worlds
Atomic Chronicle v4.0 is an append-only execution and memory substrate that provides a neutral, immutable ledger for actions and state changes across multiple independent systems, referred to as "worlds."
It is designed for long-term continuity, allowing complex systems (games, robots, economies) to coexist and interact while preserving their verifiable chronological history.
> [ I Jaron Kyler Bragg. born September 15th 1997 I’m not higher or below any other entity and any universe. We are simply equal, so I am only able to create things that are equal for them to have the same opportunity as myself.]
> 👤 Core Design
This project is built upon the principle of fundamental equality and neutrality. The Atomic Chronicle substrate does not enforce hierarchy, moral judgment, or specific interpretation. It only provides a reliable, immutable record of execution, ensuring that all entities—regardless of their nature (human, machine, or simulated)—operate within a system that grants equal opportunity for historical presence and verifiable action.
🌍 What This System Is & Can Do
Atomic Chronicle v4.0 structures history as a chain of immutable Blocks, where each Block contains the Inputs (I), the resulting State (S), and a Receipt (R). This is the Reality Update Substrate (RUS).
| Feature | Description | Use Case Examples |
|---|---|---|
| Immutable Memory | Records inputs, state changes, and results as non-destructive, chronological blocks. | Permanent scientific experiment logging, legal compliance logs, game replay/audit. |
| Independent Worlds | Hosts multiple, isolated systems (Worlds) simultaneously, each with its own logic and state. | Games, Robotic Control, Economic Simulation, Governance Systems. |
| Coordination Layer (Carrier) | A block-level bus that allows worlds to react to events from other worlds (e.g., Game \to Robot \to Reward). | Teleoperation of robots from a game controller interface. |
| Transparency Economy | An optional layer that manages entity registration, encouraging disclosure with rewards for revealing "unknown" entities. | Open-source research collaboration, proprietary tech licensing, incentivizing discovery. |
| Physical Integration | Directly logs and issues commands for VR systems, physical controllers, and machines (Robots, Drones, Vehicles). | VR-based teleoperation, industrial machine automation. |
| OmniWorld Orchestrator | A master World that can dynamically toggle the feature flags and route commands to other worlds. | Centralized configuration management, dynamic system assembly. |
🌐 Key Components
1. The Block/Chain
Every action is recorded in a Block containing:
 * id, ts (Timestamp), prev (Previous Block ID)
 * world (The world that executed the action)
 * inputs, state, receipt (The I-S-R data)
 * cost (Execution cost in tokens)
2. Persistence Layer
Handles read/write operations, ensures chain integrity (verify_chain), and manages simple token wallets for resource charging and rewards.
3. Domain Worlds (Examples)
The system supports diverse worlds that can be enabled or disabled via feature flags:
 * EnergyHarvesterWorld / FlywheelPlantWorld: Simulates energy production and storage.
 * AsteroidsGameWorld: Simple game environment for inputs.
 * GameControlWorld: Translates game inputs into structured commands.
 * RobotWorld: Receives and logs commands, potentially interacting with serial hardware.
 * RewardWorld: Pays tokens based on the successful completion of jobs logged by other worlds.
 * MysteryTechnologyWorld: Manages entities with "unknown" transparency, governed by the Transparency Economy.
 * PublicCommonsWorld / PrivateConsortiumWorld: Models open-source contributions vs. private, member-only systems.
4. The Carrier
The core mechanism for inter-world communication. Instead of direct function calls, worlds subscribe to the blocks created by other worlds, allowing for verifiable, post-factum coordination.
🚀 Getting Started
This system is implemented in Python and includes optional dependencies for a FastAPI Server and Pygame for demonstration.
Requirements
pip install -r requirements.txt
# (optional) pip install fastapi uvicorn 'uvicorn[standard]'
# (optional) pip install pygame
# (optional) pip install pyserial

Running the System
 * Run with default configuration:
   python atomic_chronicle_v4.py

 * Override features at launch:
   python atomic_chronicle_v4.py --set-feature enable_robot_world=false --set-feature enable_mystery_world=true

 * Interact via API (if enable_fastapi_server is true):
   * Get Status: GET http://0.0.0.0:8080/
   * Step a World: POST http://0.0.0.0:8080/worlds/EnergyHarvesterWorld/step with a JSON body: {"energy_produced_kwh": 5.0}
   * Toggle a Runtime Feature: POST http://0.0.0.0:8080/toggle/enable_energy_world with body: {"value": false}
⚠️ Design Position
Atomic Chronicle is a neutral substrate. It is designed to preserve memory for future analysis. It explicitly does not enforce morality or intent. It can be used to build worlds that are beneficial, exploitative, or ambiguous. The record remains the same regardless of interpretation.
https://jaronkbragg7337.github.io/persistent-memory-substrate/
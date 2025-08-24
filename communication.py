import torch
import numpy as np
from typing import Dict, List, Any
import time
import random

class NetworkSimulator:
    def __init__(self, latency_range=(0.1, 0.5), packet_loss_rate=0.01):
        self.latency_range = latency_range
        self.packet_loss_rate = packet_loss_rate
    
    def simulate_network_delay(self):
        delay = random.uniform(*self.latency_range)
        time.sleep(delay)
        return delay
    
    def simulate_packet_loss(self):
        return random.random() < self.packet_loss_rate

class Message:
    def __init__(self, sender_id: int, receiver_id: int, message_type: str, data: Any):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.message_type = message_type
        self.data = data
        self.timestamp = time.time()

class CommunicationManager:
    def __init__(self, network_simulator: NetworkSimulator = None):
        self.network_simulator = network_simulator or NetworkSimulator()
        self.message_queue = []
        self.delivered_messages = []
    
    def send_message(self, sender_id: int, receiver_id: int, message_type: str, data: Any) -> bool:
        if self.network_simulator.simulate_packet_loss():
            return False
        
        delay = self.network_simulator.simulate_network_delay()
        message = Message(sender_id, receiver_id, message_type, data)
        message.timestamp += delay
        
        self.message_queue.append(message)
        return True
    
    def receive_message(self, receiver_id: int) -> List[Message]:
        current_time = time.time()
        received_messages = []
        
        for message in self.message_queue[:]:
            if message.receiver_id == receiver_id and message.timestamp <= current_time:
                received_messages.append(message)
                self.message_queue.remove(message)
                self.delivered_messages.append(message)
        
        return received_messages
    
    def broadcast_message(self, sender_id: int, receiver_ids: List[int], message_type: str, data: Any) -> Dict[int, bool]:
        results = {}
        for receiver_id in receiver_ids:
            success = self.send_message(sender_id, receiver_id, message_type, data)
            results[receiver_id] = success
        return results

class FederatedCommunication:
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.comm_manager = CommunicationManager()
        self.server_id = -1
        self.client_ids = list(range(num_clients))
    
    def server_send_global_model(self, global_model_state: Dict[str, torch.Tensor]) -> Dict[int, bool]:
        return self.comm_manager.broadcast_message(
            self.server_id,
            self.client_ids,
            "global_model",
            global_model_state
        )
    
    def client_send_model_update(self, client_id: int, model_update: List[torch.Tensor], data_size: int) -> bool:
        update_data = {
            'model_update': model_update,
            'data_size': data_size,
            'client_id': client_id
        }
        return self.comm_manager.send_message(
            client_id,
            self.server_id,
            "model_update",
            update_data
        )
    
    def server_receive_updates(self) -> List[Dict[str, Any]]:
        messages = self.comm_manager.receive_message(self.server_id)
        updates = []
        for message in messages:
            if message.message_type == "model_update":
                updates.append(message.data)
        return updates
    
    def client_receive_global_model(self, client_id: int) -> Dict[str, torch.Tensor]:
        messages = self.comm_manager.receive_message(client_id)
        for message in messages:
            if message.message_type == "global_model":
                return message.data
        return None
    
    def get_communication_stats(self) -> Dict[str, Any]:
        total_messages = len(self.comm_manager.delivered_messages)
        failed_messages = len(self.comm_manager.message_queue)
        
        return {
            'total_messages': total_messages,
            'failed_messages': failed_messages,
            'success_rate': total_messages / (total_messages + failed_messages) if (total_messages + failed_messages) > 0 else 0
        }

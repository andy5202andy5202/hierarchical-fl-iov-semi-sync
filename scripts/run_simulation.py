from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import traci
import os
import pickle
import torch
from src.global_clock import GlobalClock
from src.global_server import GlobalServer
from src.simulation_thread import SimulationThread
from src.edge_server_init import init_edge_servers
import random


SUMO_BINARY = 'sumo'
CONFIG_FILE = ROOT_DIR / "configs" / "sumo" / "grid7x7.sumocfg"
DATA_PATH = ROOT_DIR / "cifar_non_iid"

active_training_threads = {}  # 初始化 active_training_threads
cached_node_data = {}
upload_due_to_position = {'count':0}


def preload_node_data():
    group_names = [f'g{i}' for i in range(9)]
    for group_name in group_names:
        file_path = os.path.join(DATA_PATH, f'{group_name}_train.pkl')
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    cached_node_data[group_name] = pickle.load(f)
                    print(f"預載成功：{group_name}_train.pkl")
            except Exception as e:
                print(f"[錯誤] 載入 {group_name}_train.pkl 時失敗：{e}")

def get_entry_node_from_edge(edge_id):
    """
    根據 edge_id 取得車子入口點
    例如：'n_5_5_n_5_6' → 'n_5_5'
    """
    try:
        return '_'.join(edge_id.split('_')[:3])
    except Exception as e:
        print(f"[錯誤] edge_id 解析失敗：{edge_id}, error: {e}")
        return None

def log(global_clock, message):
    """用 GlobalClock 時間標記訊息，同時寫入 env.log"""
    try:
        timestamp = f"[GlobalClock] {global_clock.get_time():.1f}s"
    except:
        timestamp = "[GlobalClock] ??s"
    
    full_msg = f"{timestamp} - {message}"
    print(full_msg,flush=True)

    # 寫入 env.log
    with open("env.log", "a") as f:
        f.write(full_msg + "\n")
        f.flush()



if __name__ == '__main__':
    if os.path.exists("env.log"):
        os.remove("env.log")
    veh_log_path = os.path.join("veh", "veh.log")
    if os.path.exists(veh_log_path):
        os.remove(veh_log_path)
    traci.start([SUMO_BINARY, '-c', CONFIG_FILE, '--collision.action', 'none'])
    preload_node_data()
    global_clock = GlobalClock()
    global_clock.start()  # 啟動global clock
    global_server = GlobalServer(global_data_path=DATA_PATH, total_edge_servers=9,upload_due_to_position=upload_due_to_position, T=120, global_clock = global_clock)
    # 定義 Edge Servers
    edge_servers = init_edge_servers(cached_node_data,DATA_PATH, active_training_threads, global_server, global_clock, upload_due_to_position)
    real_time_step = 1.0
    sim_thread = SimulationThread(step_limit=10800, real_time_step=real_time_step)
    sim_thread.start()
    global_server.start()
    
    for server in edge_servers.values():
        server.start()
        
    while sim_thread.step < 10800:
        sim_thread.step_event.wait()     # 等待模擬 step 結束
        sim_thread.step_event.clear()    # 重置
        vehicle_ids = traci.vehicle.getIDList()
        # 紀錄目前存在的車輛 ID
        existing_vehicles = set(vehicle_ids)
        
        for vid in vehicle_ids:
            try:
                if vid not in active_training_threads:
                    route = traci.vehicle.getRoute(vid)
                    if not route:
                        continue  # 確保有 route 再繼續
                    start_node = get_entry_node_from_edge(route[0])
                    # data_for_vehicle = get_data_for_vehicle(start_node)
                    active_training_threads[vid] = {
                        "entry_node": start_node,
                        "trainer": None,
                        "data_group": random.choice([f"g{i}" for i in range(9)])
                    }
                    compact_id = start_node.replace('_', '')  # 把 n_3_5_n_2_5 變成 n35
                    assigned = active_training_threads[vid]['data_group']
                    log(global_clock,f"車輛 {vid} 成功從 {compact_id} 產生並加入 active_training_threads，分配到資料 {assigned}")
            except traci.exceptions.TraCIException:
                log(global_clock,f"[錯誤] 無法取得車輛 {vid} 的位置")
                continue
            except Exception as e:
                log(global_clock,f"[未知錯誤] 處理車輛 {vid} 時發生例外：{e}")
                continue
        # 移除離開的車輛
        for vid in list(active_training_threads.keys()):
            try:
                if vid not in existing_vehicles:
                    log(global_clock,f"車輛 {vid} 離開模擬環境，移除 active_training_threads。")
                    vehicle_info = active_training_threads.pop(vid, None)
                    if vehicle_info:
                        if 'trainer' in vehicle_info and vehicle_info['trainer'] is not None:
                            vehicle_info['trainer'].stop()
                            vehicle_info['trainer'].join()
                        if 'data' in vehicle_info:
                            del vehicle_info['data']
                    torch.cuda.empty_cache()
            except Exception as e:
                log(global_clock,f"[錯誤] 移除車輛 {vid} 時發生例外：{e}")
                
    print(f"共 {upload_due_to_position['count']} 輛車是因為提前結束訓練上傳模型")          
    sim_thread.join()
    traci.close()


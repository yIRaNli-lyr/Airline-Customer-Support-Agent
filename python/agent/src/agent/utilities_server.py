# utilities_server.py (简化示例，你需要将其转换为完整的 MCP 服务器框架)

from datetime import datetime, timezone, timedelta
import requests
import json # 假设 MCP 服务器框架需要这个

# --- Tool 1: current_time ---
def current_time():
    """
    Returns the current Coordinated Universal Time (UTC) and a short example time difference.
    This helps the agent interpret relative time queries like 'tomorrow' or 'in 3 hours'.
    """
    now_utc = datetime.now(timezone.utc)
    
    # 可选增强：计算明天上午 10 点的 UTC 时间
    tomorrow = now_utc.date() + timedelta(days=1)
    tomorrow_10am = datetime.combine(tomorrow, datetime.min.time().replace(hour=10), timezone.utc)
    
    return json.dumps({
        "current_utc_time": now_utc.isoformat(),
        "tomorrow_10am_utc": tomorrow_10am.isoformat(),
        "description": "Use the 'current_utc_time' for all relative time calculations (e.g., 'tomorrow' or 'next week')."
    })

# --- Tool 2: airport_info ---
def airport_info(iata_code: str):
    """
    Retrieves basic information about an airport (e.g., city, country, IATA code) 
    by querying Wikipedia based on the IATA code. 
    Use this tool to answer general knowledge questions about airports.
    
    :param iata_code: The 3-letter IATA code of the airport (e.g., 'JFK', 'LAX').
    """
    if not iata_code:
        return json.dumps({"error": "IATA code must be provided."})
    
    # 使用 Wikipedia MediaWiki API 进行查询
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'search',
        'srsearch': f'{iata_code} airport IATA code',
        'srlimit': 1
    }
    
    try:
        response = requests.get(search_url, params=params)
        data = response.json()
        
        if data['query']['search']:
            title = data['query']['search'][0]['title']
            
            # 简化处理：我们只返回标题和 IATA 代码，实际实现中可以进行内容摘要
            return json.dumps({
                "airport_iata": iata_code.upper(),
                "wikipedia_title": title,
                "status": "Success",
                "note": "Further detail summary requires parsing the full page content."
            })
        else:
            return json.dumps({"status": "Not Found", "message": f"Could not find Wikipedia entry for IATA code {iata_code}."})

    except Exception as e:
        return json.dumps({"error": f"An API error occurred: {e}"})

# 最终需要将这些函数集成到你的 MCP 服务器启动代码中
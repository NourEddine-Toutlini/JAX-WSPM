from colorama import Fore, Back, Style, init
import pyfiglet
from colorama import Fore, Back, Style, init
import pyfiglet
from datetime import datetime
from tqdm import tqdm
import time

def print_welcome_banner():
    # Initialize colorama
    init()
    
    # Create ASCII art banner
    banner = pyfiglet.figlet_format("JAX-WSPM", font='slant')
    
    # Print decorated banner
    print(f"\n{Fore.CYAN}{banner}{Style.RESET_ALL}")
    
    # Print version and info
    print(f"{Fore.GREEN}='='='='='='='='='='='='='='='='='='='='='='='='='='='='='={Style.RESET_ALL}")
    print(f"{Fore.YELLOW}JAX-based Water and Solute Transport Process Modeling{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Version: 1.0.0{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Finite Element Implementation for Coupled Flow and Transport{Style.RESET_ALL}")
    print(f"{Fore.GREEN}='='='='='='='='='='='='='='='='='='='='='='='='='='='='='={Style.RESET_ALL}\n")
        



def print_fancy_banner():
    init()
    banner = pyfiglet.figlet_format("JAX-WSPM", font='big')
    
    # Create a box around everything
    width = 80
    box_top = f"╔{'═' * (width-2)}╗"
    box_bottom = f"╚{'═' * (width-2)}╝"
    
    print(f"\n{Fore.CYAN}{box_top}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Style.RESET_ALL}{banner.center(width-2)}{Fore.CYAN}║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{' ' * (width-2)}║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.YELLOW}{'Water and Solute Transport Modeling'.center(width-2)}{Fore.CYAN}║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{' ' * (width-2)}║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.GREEN}{' System Information '.center(width-2, '―')}║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.WHITE} • Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{' ' * 20}║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.WHITE} • Version: 1.0.0{' ' * 35}║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.WHITE} • GPU: NVIDIA A100-SXM4-40GB{' ' * 24}║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{' ' * (width-2)}║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.GREEN}{' Contact Information '.center(width-2, '―')}║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.WHITE} • Nour-eddine Toutlini{' ' * 45}║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.WHITE} • PhD student at UM6P (Maroc) / ÉTS (Montreal){' ' * 18}║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.WHITE} • noureddine.toutlini@um6p.ma{' ' * 30}║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{Fore.WHITE} • nour-eddine.toutlini.1@ens.etsmtl.ca{' ' * 18}║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║{' ' * (width-2)}║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{box_bottom}{Style.RESET_ALL}\n")

from api.alpha_vantage import fetch_stock_data

def main():
    # Call the function with the desired symbol
    stock_info = fetch_stock_data('IBM')
    print(stock_info)

if __name__ == "__main__":
    main()
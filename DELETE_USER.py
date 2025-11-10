"""
Script to delete specific users from the face recognition system
Usage: python DELETE_USER.py
"""

import os
import pandas as pd
import glob

def delete_user(user_id=None, user_name=None):
    """
    Delete a specific user from the system
    If both user_id and user_name are None, deletes all users
    """
    training_image_dir = "TrainingImage"
    employee_csv = "EmployeeDetails/EmployeeDetails.csv"
    model_file = "TrainingImageLabel/Trainer.yml"
    
    deleted_images = 0
    deleted_users = []
    
    # Delete training images
    if user_id and user_name:
        # Delete specific user
        pattern = f"{training_image_dir}/{user_name}.{user_id}.*.jpg"
        image_files = glob.glob(pattern)
        for img_file in image_files:
            try:
                os.remove(img_file)
                deleted_images += 1
            except Exception as e:
                print(f"Error deleting {img_file}: {e}")
        deleted_users.append(f"{user_name} (ID: {user_id})")
    else:
        # Delete all users
        image_files = glob.glob(f"{training_image_dir}/*.jpg")
        for img_file in image_files:
            try:
                os.remove(img_file)
                deleted_images += 1
            except Exception as e:
                print(f"Error deleting {img_file}: {e}")
        deleted_users.append("All users")
    
    # Update employee CSV
    if os.path.exists(employee_csv) and os.path.getsize(employee_csv) > 0:
        try:
            df = pd.read_csv(employee_csv)
            if user_id:
                # Remove specific user
                df = df[df['Id'] != int(user_id)]
            else:
                # Remove all users (keep header only)
                df = pd.DataFrame(columns=['Id', 'Name'])
            
            # Save updated CSV
            df.to_csv(employee_csv, index=False)
            print(f"Updated {employee_csv}")
        except Exception as e:
            print(f"Error updating CSV: {e}")
            # Create empty CSV with header
            with open(employee_csv, 'w') as f:
                f.write("Id,Name\n")
    else:
        # Create empty CSV with header
        with open(employee_csv, 'w') as f:
            f.write("Id,Name\n")
    
    # Delete trained model (must retrain after deletion)
    if os.path.exists(model_file):
        try:
            os.remove(model_file)
            print(f"Deleted trained model: {model_file}")
            print("⚠️  IMPORTANT: You must retrain the model after deleting users!")
        except Exception as e:
            print(f"Error deleting model: {e}")
    
    # Summary
    print("\n" + "="*50)
    print("DELETION SUMMARY")
    print("="*50)
    print(f"Deleted users: {', '.join(deleted_users)}")
    print(f"Deleted images: {deleted_images}")
    print(f"Employee CSV: Cleared")
    print(f"Trained model: Deleted (must retrain)")
    print("="*50)
    print("\nNext steps:")
    print("1. Run the application: python train.py")
    print("2. Add new users")
    print("3. Train the model")
    print("\n")

if __name__ == "__main__":
    import sys
    
    print("="*50)
    print("Delete Users from Face Recognition System")
    print("="*50)
    print("\nCurrent users in system:")
    
    # Show current users
    try:
        df = pd.read_csv("EmployeeDetails/EmployeeDetails.csv")
        if len(df) > 0:
            for _, row in df.iterrows():
                print(f"  - {row['Name']} (ID: {row['Id']})")
        else:
            print("  No users found")
    except:
        print("  Could not read employee database")
    
    print("\nOptions:")
    print("1. Delete ALL users")
    print("2. Delete specific user")
    print("3. Exit")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        confirm = input("\n⚠️  WARNING: This will delete ALL users! Continue? (yes/no): ").strip().lower()
        if confirm == "yes":
            delete_user()
        else:
            print("Cancelled.")
    elif choice == "2":
        user_id = input("Enter User ID to delete: ").strip()
        user_name = input("Enter User Name to delete: ").strip()
        confirm = input(f"\n⚠️  Delete user {user_name} (ID: {user_id})? (yes/no): ").strip().lower()
        if confirm == "yes":
            delete_user(user_id=user_id, user_name=user_name)
        else:
            print("Cancelled.")
    else:
        print("Exited.")


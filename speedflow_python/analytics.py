# speedflow/analytics.py
import pyds

def obj_in_roi(obj_meta) -> bool:
    user_meta_list = obj_meta.obj_user_meta_list
    while user_meta_list is not None:
        user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
        if user_meta and user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META"):
            info = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)
            if getattr(info, "roiStatus", None):
                return True
        user_meta_list = user_meta_list.next
    return False

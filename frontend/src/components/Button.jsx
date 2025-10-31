import { Link } from "react-router-dom";
const Button = ({
  isLink,
  goto,
  Icon,
  title,
  width,
  color,
  onClickFun,
  hoverStyles,
  linkState,
  iconColor,
  textColor,
}) => {
  return isLink ? (
    <Link
      to={goto}
      state={linkState}
      className={`flex justify-center ${Icon && title ? "gap-1" : ""} ${
        width && `w-[${width}]`
      } items-center cursor-pointer px-3 py-3 ${color}  ${hoverStyles} transition-all text-center rounded-[4px]`}
    >
      {Icon && <span>{Icon}</span>}
      <span className={`text-${textColor}`}>{title}</span>{" "}
    </Link>
  ) : (
    <div
      onClick={onClickFun}
      className={`flex justify-center ${Icon && title ? "gap-1" : ""} ${
        width && `w-[${width}]`
      } items-center cursor-pointer px-3 py-3 ${color} ${hoverStyles} transition-all text-center rounded-[4px]`}
    >
      {Icon && <span className={`text-${iconColor}`}>{Icon}</span>}
      <span className={`text-${textColor}`}>{title}</span>{" "}
    </div>
  );
};

export default Button;
